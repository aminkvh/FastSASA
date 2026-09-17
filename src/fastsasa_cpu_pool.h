/* Persistent worker threads for the CPU kernels' per-call parallel loops.
 *
 * Spawning a thread costs ~20 us, so launching one per core on every call
 * put a ~0.3 ms floor under small structures and ~10% on a 25k-atom frame.
 * Parked workers wake in a few microseconds. Only one run() uses the pool
 * at a time; a caller that finds it busy (nested or concurrent contexts),
 * or that runs in a forked child, spawns threads as before, so the
 * concurrency semantics of the public API are unchanged. */
#ifndef FASTSASA_CPU_POOL_H
#define FASTSASA_CPU_POOL_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace fastsasa_cpu {

class WorkerPool {
public:
    static WorkerPool &instance()
    {
        static WorkerPool pool;
        return pool;
    }

    /* Runs task(0) on the calling thread and task(1 .. count-1) on the
     * workers, returning when every task has finished. task must not throw.
     * The workers are started on the first call that needs them. */
    static void run(int count, const std::function<void(int)> &task)
    {
        if (count <= 1) {
            task(0);
            return;
        }
        instance().execute(count, task);
    }

private:
    void execute(int count, const std::function<void(int)> &task)
    {
        if (count - 1 > static_cast<int>(workers_.size()) || !owns_process() ||
            busy_.exchange(true, std::memory_order_acq_rel)) {
            run_spawned(count, task);
            return;
        }
        task_ = &task;
        count_ = count;
        pending_.store(static_cast<int>(workers_.size()), std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            generation_.fetch_add(1, std::memory_order_acq_rel);
        }
        wake_.notify_all();
        task(0);
        wait_pending();
        task_ = nullptr;
        busy_.store(false, std::memory_order_release);
    }

    WorkerPool()
    {
        const unsigned available = std::thread::hardware_concurrency();
        const int n = available > 1u ? static_cast<int>(available - 1u) : 0;
#ifndef _WIN32
        owner_ = getpid();
#endif
        workers_.reserve(static_cast<size_t>(n));
        try {
            for (int i = 0; i < n; ++i) workers_.emplace_back(&WorkerPool::worker_loop, this, i);
        } catch (...) {
            /* Whatever was started keeps serving; run() spawns for the rest. */
        }
    }

    ~WorkerPool()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            generation_.fetch_add(1, std::memory_order_acq_rel);
        }
        wake_.notify_all();
        for (std::thread &worker : workers_) {
#ifdef _WIN32
            /* Process exit has already ended the workers; joining from a
             * static destructor can deadlock on the loader lock. */
            worker.detach();
#else
            worker.join();
#endif
        }
    }

    bool owns_process() const
    {
#ifndef _WIN32
        return getpid() == owner_;
#else
        return true;
#endif
    }

    /* Workers spin briefly before sleeping so back-to-back calls (a
     * trajectory loop) never pay a futex wake. */
    static bool spin_until(const std::function<bool()> &ready)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(50);
        while (std::chrono::steady_clock::now() < deadline) {
            if (ready()) return true;
        }
        return ready();
    }

    void worker_loop(int index)
    {
        std::uint64_t seen = 0;
        for (;;) {
            if (!spin_until([&] { return generation_.load(std::memory_order_acquire) != seen; })) {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&] { return generation_.load(std::memory_order_acquire) != seen; });
            }
            seen = generation_.load(std::memory_order_acquire);
            if (stop_) return;
            if (index + 1 < count_) (*task_)(index + 1);
            if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(mutex_);
                done_.notify_all();
            }
        }
    }

    void wait_pending()
    {
        if (spin_until([&] { return pending_.load(std::memory_order_acquire) == 0; })) return;
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [&] { return pending_.load(std::memory_order_acquire) == 0; });
    }

    static void run_spawned(int count, const std::function<void(int)> &task)
    {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(count - 1));
        try {
            for (int i = 1; i < count; ++i) threads.emplace_back([&task, i]() { task(i); });
        } catch (...) {
            for (std::thread &thread : threads) thread.join();
            throw;
        }
        task(0);
        for (std::thread &thread : threads) thread.join();
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    std::atomic<std::uint64_t> generation_{0};
    std::atomic<int> pending_{0};
    std::atomic<bool> busy_{false};
    bool stop_ = false;
    const std::function<void(int)> *task_ = nullptr;
    int count_ = 0;
#ifndef _WIN32
    pid_t owner_ = 0;
#endif
};

} // namespace fastsasa_cpu

#endif
