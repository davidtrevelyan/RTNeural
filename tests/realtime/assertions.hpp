#pragma once

#ifdef RTNEURAL_RADSAN_ENABLED

extern "C" void radsan_realtime_enter();
extern "C" void radsan_realtime_exit();
#define EXPECT_REAL_TIME_SAFE(statement) \
    radsan_realtime_enter();             \
    statement;                           \
    radsan_realtime_exit();
#else
#define EXPECT_REAL_TIME_SAFE(statement) \
    #error EXPECT_REAL_TIME_SAFE         \
        requires RTNEURAL_RADSAN_ENABLED
#endif
