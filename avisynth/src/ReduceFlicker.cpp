/*
ReduceFlicker.cpp

This file is a part of ReduceFlicker.

Copyright (C) 2016 OKA Motofumi

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
*/


#include <algorithm>

#include "ReduceFlicker.h"
#include "avs/minmax.h"

extern proc_filter_t get_main_proc(arch_t arch, int strength, bool aggressive, int bits_per_sample);

static void copy_plane(PVideoFrame& dst, PVideoFrame& src, int plane, IScriptEnvironment* env)
{
    const uint8_t* srcp = src->GetReadPtr(plane);
    int src_pitch = src->GetPitch(plane);
    int height = src->GetHeight(plane);
    int row_size = src->GetRowSize(plane);
    uint8_t* destp = dst->GetWritePtr(plane);
    int dst_pitch = dst->GetPitch(plane);
    env->BitBlt(destp, dst_pitch, srcp, src_pitch, row_size, height);
}

ReduceFlicker::ReduceFlicker(PClip c, int s, bool aggressive, bool grey, arch_t arch, bool ra, bool luma, IScriptEnvironment* env) :
    GenericVideoFilter(c), strength(s), _grey(grey), raccess(ra), _luma(luma)
{
    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; }

    int planecount = min(vi.NumComponents(), 3);
    for (int i = 0; i < planecount; i++)
    {
        if (i == 0)
            processPlane[i] = _luma;
        else
            processPlane[i] = !_grey;
    }

    align = arch == arch_t::USE_AVX2 ? 32 : 16;
    
    int bits_per_sample;
    if (vi.BitsPerComponent() == 8)
        bits_per_sample = 8;
    else if (vi.BitsPerComponent() <= 16)
        bits_per_sample = 16;
    else
        bits_per_sample = 32;

    mainProc = get_main_proc(arch, strength, aggressive, bits_per_sample);
}


PVideoFrame __stdcall ReduceFlicker::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame curr, prev[3], next[3];
    const int nf = vi.num_frames - 1;

    if (raccess) {
        switch (strength) {
        case 3:
            next[2] = child->GetFrame(std::min(n + 3, nf), env);
            next[1] = child->GetFrame(std::min(n + 2, nf), env);
            next[0] = child->GetFrame(std::min(n + 1, nf), env);
            curr = child->GetFrame(n, env);
            prev[0] = child->GetFrame(std::max(n - 1, 0), env);
            prev[1] = child->GetFrame(std::max(n - 2, 0), env);
            prev[2] = child->GetFrame(std::max(n - 3, 0), env);
            break;
        case 2:
            next[1] = child->GetFrame(std::min(n + 2, nf), env);
            next[0] = child->GetFrame(std::min(n + 1, nf), env);
            curr = child->GetFrame(n, env);
            prev[0] = child->GetFrame(std::max(n - 1, 0), env);
            prev[1] = child->GetFrame(std::max(n - 2, 0), env);
            break;
        default:
            next[0] = child->GetFrame(std::min(n + 1, nf), env);
            curr = child->GetFrame(n, env);
            prev[1] = child->GetFrame(std::max(n - 2, 0), env);
            prev[0] = child->GetFrame(std::max(n - 1, 0), env);
        }
    } else {
        switch (strength) {
        case 3:
            prev[2] = child->GetFrame(std::max(n - 3, 0), env);
            prev[1] = child->GetFrame(std::max(n - 2, 0), env);
            prev[0] = child->GetFrame(std::max(n - 1, 0), env);
            curr = child->GetFrame(n, env);
            next[0] = child->GetFrame(std::min(n + 1, nf), env);
            next[1] = child->GetFrame(std::min(n + 2, nf), env);
            next[2] = child->GetFrame(std::min(n + 3, nf), env);
            break;
        case 2:
            prev[1] = child->GetFrame(std::max(n - 2, 0), env);
            prev[0] = child->GetFrame(std::max(n - 1, 0), env);
            curr = child->GetFrame(n, env);
            next[0] = child->GetFrame(std::min(n + 1, nf), env);
            next[1] = child->GetFrame(std::min(n + 2, nf), env);
            break;
        default:
            prev[1] = child->GetFrame(std::max(n - 2, 0), env);
            prev[0] = child->GetFrame(std::max(n - 1, 0), env);
            curr = child->GetFrame(n, env);
            next[0] = child->GetFrame(std::min(n + 1, nf), env);
        }
    }

    PVideoFrame dst;
    if (has_at_least_v8) dst = env->NewVideoFrameP(vi, &curr, align); else dst = env->NewVideoFrame(vi, align);

    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    const int* current_planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
    int planecount = std::min(vi.NumComponents(), 3);
    for (int i = 0; i < planecount; i++)
    {
        const int plane = current_planes[i];

        if (processPlane[i])
        {
            int width = curr->GetRowSize(plane);
            int height = curr->GetHeight(plane);
            int cpitch = curr->GetPitch(plane);
            int dpitch = dst->GetPitch(plane);
            const uint8_t* currp = curr->GetReadPtr(plane);
            uint8_t* dstp = dst->GetWritePtr(plane);

            const uint8_t* prevp[3], * nextp[3];
            int ppitch[3], npitch[3];
            switch (strength) {
            case 3:
                prevp[2] = prev[2]->GetReadPtr(plane);
                ppitch[2] = prev[2]->GetPitch(plane);
                nextp[2] = next[2]->GetReadPtr(plane);
                npitch[2] = next[2]->GetPitch(plane);
            case 2:
                nextp[1] = next[1]->GetReadPtr(plane);
                npitch[1] = next[1]->GetPitch(plane);
            default:
                prevp[0] = prev[0]->GetReadPtr(plane);
                ppitch[0] = prev[0]->GetPitch(plane);
                prevp[1] = prev[1]->GetReadPtr(plane);
                ppitch[1] = prev[1]->GetPitch(plane);
                nextp[0] = next[0]->GetReadPtr(plane);
                npitch[0] = next[0]->GetPitch(plane);
            }

            mainProc(dstp, currp, prevp, nextp, dpitch, cpitch, ppitch, npitch,
                width / vi.ComponentSize(), height);
        }
    }
    
    return dst;
}

int __stdcall ReduceFlicker::SetCacheHints(int cachehints, int frame_range)
{
    return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
}

static arch_t get_arch(int opt, IScriptEnvironment* env) noexcept
{
    const bool has_sse2 = env->GetCPUFlags() & CPUF_SSE2;
    const bool has_sse41 = env->GetCPUFlags() & CPUF_SSE4_1;
    const bool has_avx2 = env->GetCPUFlags() & CPUF_AVX2;

#if !defined(__SSE2__)
    return NO_SIMD
#else
    if (opt == 0 || !has_sse2)
        return arch_t::NO_SIMD;

#if !defined(__SSE4_1__)
    return USE_SSE2;
#else
    if (opt == 1 || !has_sse41)
        return arch_t::USE_SSE2;

#if !defined(__AVX2__)
    return USE_SSE41;
#else
    if (opt == 2 || !has_avx2)
        return arch_t::USE_SSE41;

    return arch_t::USE_AVX2;
#endif // __AVX2__
#endif // __SSE4_1__
#endif // __SSE2__
}

static void validate(bool cond, const char* msg)
{
    if (cond) throw msg;
}

AVSValue __cdecl ReduceFlicker::create(AVSValue args, void*, IScriptEnvironment* env)
{
    try {
        PClip clip = args[0].AsClip();
        const VideoInfo& vi = clip->GetVideoInfo();
        validate(vi.IsRGB(), "input clip must be in Y/YUV 8..32-bit format.");

        int strength = args[1].AsInt(2);
        validate(strength < 1 || strength > 3,
                 "strength must be set to 1, 2 or 3.");
        
        bool aggressive = args[2].AsBool(false);
        bool grey = args[3].AsBool(false);

        arch_t arch = get_arch(args[4].AsInt(-1), env);
        validate(args[4].AsInt() < 0 || args[4].AsInt() > 3,
            "opt must be between 0..3.");

        bool raccess = args[5].AsBool(true);

        bool luma = args[6].AsBool(true);
        return new ReduceFlicker(clip, strength, aggressive, grey, arch, raccess, luma, env);

    } catch (const char* e) {
        env->ThrowError("ReduceFlicker: %s", e);
    }
    return 0;
}


const AVS_Linkage* AVS_linkage = nullptr;


extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;
    env->AddFunction("ReduceFlicker",
                     "c[strength]i[aggressive]b[grey]b[opt]i[raccess]b[luma]b",
                     ReduceFlicker::create, nullptr);

    return "ReduceFlicker for avs2.6/avs+ ver. " REDUCE_FLICKER_VERSION;
}
