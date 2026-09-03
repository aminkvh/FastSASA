/*
 * Small portability layer for the few OS services FastSASA needs outside
 * the C standard library: a mutex, a joinable thread, setting an
 * environment variable, and a monotonic clock. POSIX systems map onto
 * pthreads; Windows maps onto SRW locks, CreateThread, _putenv_s and
 * QueryPerformanceCounter. Internal header, not part of the public API.
 */
#ifndef FASTSASA_PORTABLE_H
#define FASTSASA_PORTABLE_H

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

/* 64-bit file offsets: MSVC has no fseeko/ftello, only _fseeki64/_ftelli64. */
typedef __int64 fastsasa_file_offset;
#define fastsasa_fseek(file, offset, whence) _fseeki64((file), (offset), (whence))
#define fastsasa_ftell(file) _ftelli64(file)

typedef SRWLOCK fastsasa_mutex;
#define FASTSASA_MUTEX_INIT SRWLOCK_INIT

static inline void fastsasa_mutex_lock(fastsasa_mutex *mutex) { AcquireSRWLockExclusive(mutex); }
static inline void fastsasa_mutex_unlock(fastsasa_mutex *mutex) { ReleaseSRWLockExclusive(mutex); }

typedef struct {
    HANDLE handle;
    void *(*function)(void *);
    void *argument;
} fastsasa_thread;

static DWORD WINAPI
fastsasa_thread_trampoline(LPVOID parameter)
{
    fastsasa_thread *thread = (fastsasa_thread *)parameter;

    thread->function(thread->argument);
    return 0;
}

/* The fastsasa_thread object must stay alive until fastsasa_thread_join. */
static inline int
fastsasa_thread_create(fastsasa_thread *thread, void *(*function)(void *), void *argument)
{
    thread->function = function;
    thread->argument = argument;
    thread->handle = CreateThread(NULL, 0, fastsasa_thread_trampoline, thread, 0, NULL);
    return thread->handle != NULL ? 0 : -1;
}

static inline int
fastsasa_thread_join(fastsasa_thread *thread)
{
    const DWORD waited = WaitForSingleObject(thread->handle, INFINITE);

    CloseHandle(thread->handle);
    thread->handle = NULL;
    return waited == WAIT_OBJECT_0 ? 0 : -1;
}

static inline int
fastsasa_setenv(const char *name, const char *value)
{
    return _putenv_s(name, value) == 0 ? 0 : -1;
}

static inline double
fastsasa_monotonic_seconds(void)
{
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}

#else /* POSIX */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>

/* Callers define _FILE_OFFSET_BITS 64 before any include so off_t is 64-bit. */
typedef off_t fastsasa_file_offset;
#define fastsasa_fseek(file, offset, whence) fseeko((file), (offset), (whence))
#define fastsasa_ftell(file) ftello(file)

typedef pthread_mutex_t fastsasa_mutex;
#define FASTSASA_MUTEX_INIT PTHREAD_MUTEX_INITIALIZER

static inline void fastsasa_mutex_lock(fastsasa_mutex *mutex) { pthread_mutex_lock(mutex); }
static inline void fastsasa_mutex_unlock(fastsasa_mutex *mutex) { pthread_mutex_unlock(mutex); }

typedef pthread_t fastsasa_thread;

static inline int
fastsasa_thread_create(fastsasa_thread *thread, void *(*function)(void *), void *argument)
{
    return pthread_create(thread, NULL, function, argument) == 0 ? 0 : -1;
}

static inline int
fastsasa_thread_join(fastsasa_thread *thread)
{
    return pthread_join(*thread, NULL) == 0 ? 0 : -1;
}

static inline int
fastsasa_setenv(const char *name, const char *value)
{
    return setenv(name, value, 1);
}

static inline double
fastsasa_monotonic_seconds(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#endif

#endif /* FASTSASA_PORTABLE_H */
