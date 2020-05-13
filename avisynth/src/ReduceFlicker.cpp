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
#include <map>
#include <tuple>

#include "ReduceFlicker.h"
#include "proc_filter.h"




proc_filter_t
get_main_proc(int strength, bool aggressive, arch_t arch)
{
    using std::make_tuple;

    std::map<std::tuple<int, bool, arch_t>, proc_filter_t> func;

    func[make_tuple(1, false, arch_t::NO_SIMD)] = proc_c<1>;
    func[make_tuple(2, false, arch_t::NO_SIMD)] = proc_c<2>;
    func[make_tuple(3, false, arch_t::NO_SIMD)] = proc_c<3>;

    func[make_tuple(1, true, arch_t::NO_SIMD)] = proc_a_c<1>;
    func[make_tuple(2, true, arch_t::NO_SIMD)] = proc_a_c<2>;
    func[make_tuple(3, true, arch_t::NO_SIMD)] = proc_a_c<3>;

    func[make_tuple(1, false, arch_t::USE_SSE2)] = proc_simd<__m128i, 1>;
    func[make_tuple(2, false, arch_t::USE_SSE2)] = proc_simd<__m128i, 2>;
    func[make_tuple(3, false, arch_t::USE_SSE2)] = proc_simd<__m128i, 3>;

    func[make_tuple(1, true, arch_t::USE_SSE2)] = proc_a_simd<__m128i, 1>;
    func[make_tuple(2, true, arch_t::USE_SSE2)] = proc_a_simd<__m128i, 2>;
    func[make_tuple(3, true, arch_t::USE_SSE2)] = proc_a_simd<__m128i, 3>;
#if defined(__AVX2__)
    func[make_tuple(1, false, arch_t::USE_AVX2)] = proc_simd<__m256i, 1>;
    func[make_tuple(2, false, arch_t::USE_AVX2)] = proc_simd<__m256i, 2>;
    func[make_tuple(3, false, arch_t::USE_AVX2)] = proc_simd<__m256i, 3>;

    func[make_tuple(1, true, arch_t::USE_AVX2)] = proc_a_simd<__m256i, 1>;
    func[make_tuple(2, true, arch_t::USE_AVX2)] = proc_a_simd<__m256i, 2>;
    func[make_tuple(3, true, arch_t::USE_AVX2)] = proc_a_simd<__m256i, 3>;
#endif
    return func[make_tuple(strength, aggressive, arch)];
}



ReduceFlicker::
ReduceFlicker(PClip c, int s, bool aggressive, bool grey, arch_t arch, bool ra, ise_t* env) :
    GenericVideoFilter(c), strength(s), raccess(ra)
{
    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; }
    numPlanes = vi.IsY8() || grey ? 1 : 3;
    align = arch == arch_t::USE_AVX2 ? 32 : 16;
    mainProc = get_main_proc(strength, aggressive, arch);
}


PVideoFrame __stdcall ReduceFlicker::GetFrame(int n, ise_t* env)
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
            next[1] = child->GetFrame(std::min(n + 1, nf), env);
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

    const int planes[] = {PLANAR_Y, PLANAR_U, PLANAR_V};
    for (int p = 0; p < numPlanes; ++p) {
        const int plane = planes[p];
        
        const int width = curr->GetRowSize(plane);
        const int height = curr->GetHeight(plane);
        const int cpitch = curr->GetPitch(plane);
        const int dpitch = dst->GetPitch(plane);
        const uint8_t* currp = curr->GetReadPtr(plane);
        uint8_t* dstp = dst->GetWritePtr(plane);
        
        const uint8_t *prevp[3], *nextp[3];
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

        mainProc(dstp, prevp, currp, nextp, dpitch, ppitch, cpitch, npitch,
                 width, height);
    }
    
    return dst;
}

int __stdcall ReduceFlicker::SetCacheHints(int cachehints, int frame_range)
{
    return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
}

static arch_t get_arch(int opt, ise_t* env) noexcept
{
    const bool has_sse2 = env->GetCPUFlags() & CPUF_SSE2;
    const bool has_avx2 = env->GetCPUFlags() & CPUF_AVX2;

    if (opt == static_cast<int>(arch_t::NO_SIMD) || !has_sse2) {
        return arch_t::NO_SIMD;
    }
#if !defined(__AVX2__)
    return USE_SSE2;
#else
    if (opt == static_cast<int>(arch_t::USE_SSE2) || !has_avx2) {
        return arch_t::USE_SSE2;
    }
    return arch_t::USE_AVX2;
#endif // __AVX2__
}

static void validate(bool cond, const char* msg)
{
    if (cond) throw msg;
}

AVSValue __cdecl ReduceFlicker::create(AVSValue args, void*, ise_t* env)
{
    try {
        PClip clip = args[0].AsClip();
        const VideoInfo& vi = clip->GetVideoInfo();
        validate(!vi.IsPlanar(), "input clip must be planar format.");

        int strength = args[1].AsInt(2);
        validate(strength < 1 || strength > 3,
                 "strength must be set to 1, 2 or 3.");
        
        bool aggressive = args[2].AsBool(false);
        bool grey = args[3].AsBool(false);

        bool is_avsplus = env->FunctionExists("SetFilterMTMode");
        arch_t arch = get_arch(args[4].AsInt(static_cast<int>(arch_t::USE_AVX2)), env);
        if (arch == arch_t::USE_AVX2 && !is_avsplus) {
            arch = arch_t::USE_SSE2;
        }

        bool raccess = args[5].AsBool(true);
        return new ReduceFlicker(clip, strength, aggressive, grey, arch, raccess, env);

    } catch (const char* e) {
        env->ThrowError("ReduceFlicker: %s", e);
    }
    return 0;
}


const AVS_Linkage* AVS_linkage = nullptr;


extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(ise_t* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;
    env->AddFunction("ReduceFlicker",
                     "c[strength]i[aggressive]b[grey]b[opt]i[raccess]b",
                     ReduceFlicker::create, nullptr);

    return "ReduceFlicker for avs2.6/avs+ ver. " REDUCE_FLICKER_VERSION;
}
