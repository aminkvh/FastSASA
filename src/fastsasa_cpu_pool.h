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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace fastsasa_cpu {

class WorkerPool {
public:
    /* Never destroyed: a static destructor would tear down the mutex and
     * condition variables while a worker can still be waking on them
     * (observed as a crash during exit on Windows, truncating stdout). The
     * parked workers cost nothing and the OS reclaims them at exit. */
    static WorkerPool &instance()
    {
        static WorkerPool *pool = new WorkerPool(default_workers());
        return *pool;
    }

    /* A private pool of `workers` threads that run() uses on the creating
     * thread while the Team exists. Concurrent trajectory frames each get a
     * team: through the shared pool only one frame at a time could split its
     * work, and the others would spawn threads on every call. */
    class Team {
    public:
        explicit Team(int workers) : previous_(current())
        {
            if (workers <= 0) return;
            try {
                pool_.reset(new WorkerPool(workers));
            } catch (...) {
                return; /* run() falls back to the shared pool or to spawning */
            }
            current() = pool_.get();
        }
        ~Team()
        {
            if (!pool_) return;
            current() = previous_;
            pool_->shutdown();
        }
        Team(const Team &) = delete;
        Team &operator=(const Team &) = delete;

    private:
        WorkerPool *previous_;
        std::unique_ptr<WorkerPool> pool_;
    };

    /* Runs task(0) on the calling thread and task(1 .. count-1) on the
     * workers, returning when every task has finished. task must not throw.
     * The workers are started on the first call that needs them. */
    static void run(int count, const std::function<void(int)> &task)
    {
        if (count <= 1) {
            task(0);
            return;
        }
        WorkerPool *team = current();
        (team != nullptr ? *team : instance()).execute(count, task);
    }

    /* Hands [0, n) to count tasks in chunks taken from a shared counter, so
     * a task that finishes early takes more work instead of idling at the
     * join; body(begin, end, task) runs once per chunk. Every caller computes
     * each item independently of the others, so the split never changes
     * results. */
    template <class Body>
    static void run_chunked(int count, int n, const Body &body)
    {
        if (n <= 0) return;
        const int chunk = chunk_size(count, n);
        std::atomic<int> next(0);
        run(count, [&](int task) {
            for (;;) {
                const int begin = next.fetch_add(chunk, std::memory_order_relaxed);
                if (begin >= n) break;
                body(begin, std::min(n, begin + chunk), task);
            }
        });
    }

    /* About 16 chunks per task evens out the load (surface and buried atoms
     * differ several-fold in cost); 16 to 256 items keeps taking a chunk
     * negligible next to the work in it. */
    static int chunk_size(int count, int n)
    {
        if (count <= 1) return std::max(1, n);
        return std::max(16, std::min(256, n / (count * 16)));
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

    static WorkerPool *&current()
    {
        static thread_local WorkerPool *team = nullptr;
        return team;
    }

    static int default_workers()
    {
        const unsigned available = std::thread::hardware_concurrency();
        return available > 1u ? static_cast<int>(available - 1u) : 0;
    }

    explicit WorkerPool(int n)
    {
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

    /* Teams only: called when no run() is in progress. */
    void shutdown()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_.store(true, std::memory_order_release);
        }
        wake_.notify_all();
        for (std::thread &worker : workers_) worker.join();
    }

    void worker_loop(int index)
    {
        std::uint64_t seen = 0;
        for (;;) {
            const auto ready = [&] {
                return stop_.load(std::memory_order_acquire) || generation_.load(std::memory_order_acquire) != seen;
            };
            if (!spin_until(ready)) {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, ready);
            }
            if (stop_.load(std::memory_order_acquire)) return;
            seen = generation_.load(std::memory_order_acquire);
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
    std::atomic<bool> stop_{false};
    const std::function<void(int)> *task_ = nullptr;
    int count_ = 0;
#ifndef _WIN32
    pid_t owner_ = 0;
#endif
};

} // namespace fastsasa_cpu

#endif
