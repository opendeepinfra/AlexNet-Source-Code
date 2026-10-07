/*
 * Minimal stand-in for the CUDA SDK samples' helper_timer.h.
 *
 * The 2012 release includes <helper_timer.h> from include/layer.cuh, but every
 * actual use of the timer is either commented out or lives in src/test.cu
 * (a standalone unit-test program that is not part of the Python extension).
 * A small self-contained implementation is provided so the header resolves and
 * the API stays available for anyone who uncomments the timing code.
 */

#ifndef COMMON_HELPER_TIMER_H_
#define COMMON_HELPER_TIMER_H_

#include <time.h>

struct StopWatchInterface {
    double startTime;
    double totalTime;
    int running;
};

static inline double sdkTimersNow(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1000.0 + (double) ts.tv_nsec / 1000000.0;
}

static inline void sdkCreateTimer(StopWatchInterface **timer_interface) {
    *timer_interface = new StopWatchInterface();
    (*timer_interface)->startTime = 0.0;
    (*timer_interface)->totalTime = 0.0;
    (*timer_interface)->running = 0;
}

static inline void sdkDeleteTimer(StopWatchInterface **timer_interface) {
    if (*timer_interface) {
        delete *timer_interface;
        *timer_interface = NULL;
    }
}

static inline void sdkResetTimer(StopWatchInterface **timer_interface) {
    (*timer_interface)->totalTime = 0.0;
    (*timer_interface)->running = 0;
}

static inline void sdkStartTimer(StopWatchInterface **timer_interface) {
    (*timer_interface)->startTime = sdkTimersNow();
    (*timer_interface)->running = 1;
}

static inline void sdkStopTimer(StopWatchInterface **timer_interface) {
    if ((*timer_interface)->running) {
        (*timer_interface)->totalTime += sdkTimersNow() - (*timer_interface)->startTime;
        (*timer_interface)->running = 0;
    }
}

/* Elapsed time in milliseconds since the last sdkResetTimer(). */
static inline float sdkGetTimerValue(StopWatchInterface **timer_interface) {
    double total = (*timer_interface)->totalTime;
    if ((*timer_interface)->running) {
        total += sdkTimersNow() - (*timer_interface)->startTime;
    }
    return (float) total;
}

#endif  /* COMMON_HELPER_TIMER_H_ */
