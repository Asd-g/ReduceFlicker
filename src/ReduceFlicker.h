#pragma once

#include "avisynth.h"
#include "avs/minmax.h"

#ifdef _MSC_VER
#define F_INLINE __forceinline
#elif defined(GCC) || defined(CLANG)
#define F_INLINE inline __attribute__((__always_inline__))
#endif

template <typename T, int STRENGTH>
void proc_sse2(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template <typename T, int STRENGTH>
void proc_a_sse2(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;

template <typename T, int STRENGTH>
void proc_avx2(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template <typename T, int STRENGTH>
void proc_a_avx2(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;

class ReduceFlicker : public GenericVideoFilter
{
    int strength;
    bool _grey;
    int opt_;
    size_t align;
    bool raccess, _luma;
    bool processPlane[3];
    bool has_at_least_v8;
    bool avx2, sse2;

    void (*process)(uint8_t*, const uint8_t*, const uint8_t**, const uint8_t**, int, int, int*, int*, int, int) noexcept;

public:
    ReduceFlicker(PClip c, int str, bool agr, bool grey, int opt, bool raccess, bool luma, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
    int __stdcall SetCacheHints(int cachehints, int frame_range)
    {
        return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
};
