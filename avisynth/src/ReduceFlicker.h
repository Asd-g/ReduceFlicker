#ifndef REDUCE_FLICKER_H
#define REDUCE_FLICKER_H

#include <cstdint>

#include "avisynth.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#define REDUCE_FLICKER_VERSION "0.1.0"

#define __SSE2__
#define __SSE4_1__
#define __AVX2__

enum class arch_t {
    NO_SIMD,
    USE_SSE2,
    USE_SSE41,
    USE_AVX2
};

typedef void(__stdcall *proc_filter_t)(
    uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp,
    const uint8_t** nextp, int dstride, int cstride, int* pstride,
    int* nstride, int width, int height);

class ReduceFlicker : public GenericVideoFilter {
    const int strength;
    bool _grey, _luma;
    size_t align;
    bool raccess;
    bool processPlane[3];
    bool has_at_least_v8;

    proc_filter_t mainProc;

public:
    ReduceFlicker(PClip c, int str, bool agr, bool grey, arch_t arch, bool raccess, bool luma, IScriptEnvironment* env);
    ~ReduceFlicker() {}
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
    static AVSValue __cdecl create(AVSValue args, void*, IScriptEnvironment* env);
    int __stdcall SetCacheHints(int cachehints, int frame_range);
};

#endif
