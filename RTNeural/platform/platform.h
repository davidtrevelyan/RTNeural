#pragma once

#ifdef RTNEURAL_RADSAN_ENABLED
  #define RTNEURAL_RT_ATTR [[clang::realtime]]
#else
  #define RTNEURAL_RT_ATTR
#endif
