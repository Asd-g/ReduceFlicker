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

#include "ReduceFlicker.h"

template <typename T>
static F_INLINE T absdiff(const T x, const T y)
{
    return x < y ? y - x : x - y;
}

template <typename T>
static F_INLINE T get_avg(T a, T b, T x)
{
    int t = max((a + b + 1) / 2 - 1, 0);
    return static_cast<T>((t + x + 1) / 2);
}

template <>
F_INLINE float get_avg(float a, float b, float x)
{
    return (a + b + x + x) * 0.25f;
}

template <typename T0, int STRENGTH>
static void proc_c(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept
{
    using T1 = std::conditional_t<std::is_integral_v<T0>, int, float>;

    const T0* prv0, * prv1, * prv2, * nxt0, * nxt1, * nxt2;

    T0* dst0 = reinterpret_cast<T0*>(dstp);
    const T0* cur0 = reinterpret_cast<const T0*>(currp);
    prv0 = reinterpret_cast<const T0*>(prevp[0]);
    prv1 = reinterpret_cast<const T0*>(prevp[1]);
    nxt0 = reinterpret_cast<const T0*>(nextp[0]);
    dstride /= sizeof(T0);
    cstride /= sizeof(T0);
    pstride[0] /= sizeof(T0);
    pstride[1] /= sizeof(T0);
    nstride[0] /= sizeof(T0);
    if (STRENGTH > 1)
    {
        nxt1 = reinterpret_cast<const T0*>(nextp[1]);
        nstride[1] /= sizeof(T0);
    }
    if (STRENGTH > 2)
    {
        prv2 = reinterpret_cast<const T0*>(prevp[2]);
        nxt2 = reinterpret_cast<const T0*>(nextp[2]);
        pstride[2] /= sizeof(T0);
        nstride[2] /= sizeof(T0);
    }

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const T1 curx = static_cast<T1>(cur0[x]);
            T1 d = absdiff(curx, static_cast<T1>(prv1[x]));
            if (STRENGTH > 1)
            {
                d = min(d, absdiff(curx, static_cast<T1>(nxt1[x])));
            }
            if (STRENGTH > 2)
            {
                d = min(d, absdiff(curx, static_cast<T1>(prv2[x])));
                d = min(d, absdiff(curx, static_cast<T1>(nxt2[x])));
            }
            T1 prvx = static_cast<T1>(prv0[x]);
            T1 nxtx = static_cast<T1>(nxt0[x]);
            T1 avg = get_avg(prvx, nxtx, curx);
            T1 ul = max(min(prvx, nxtx) - d, curx);
            T1 ll = min(max(prvx, nxtx) + d, curx);
            dst0[x] = static_cast<T0>(clamp(avg, ll, ul));
        }
        dst0 += dstride;
        cur0 += cstride;
        prv0 += pstride[0];
        prv1 += pstride[1];
        nxt0 += nstride[0];
        if (STRENGTH > 1)
        {
            nxt1 += nstride[1];
        }
        if (STRENGTH > 2)
        {
            prv2 += pstride[2];
            nxt2 += nstride[2];
        }
    }
}

template <typename T>
static F_INLINE void update_diff(T x, T y, T& d1, T& d2)
{
    T d = x - y;
    if (d >= 0)
    {
        d2 = 0;
        d1 = min(d, d1);
    }
    else
    {
        d1 = 0;
        d2 = min(-d, d2);
    }
}

template <typename T0, int STRENGTH>
static void proc_a_c(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept
{
    using T1 = std::conditional_t<std::is_integral_v<T0>, int, float>;

    const T0* prv0, * prv1, * prv2, * nxt0, * nxt1, * nxt2;

    T0* dst0 = reinterpret_cast<T0*>(dstp);
    const T0* cur0 = reinterpret_cast<const T0*>(currp);
    prv0 = reinterpret_cast<const T0*>(prevp[0]);
    prv1 = reinterpret_cast<const T0*>(prevp[1]);
    nxt0 = reinterpret_cast<const T0*>(nextp[0]);
    dstride /= sizeof(T0);
    cstride /= sizeof(T0);
    pstride[0] /= sizeof(T0);
    pstride[1] /= sizeof(T0);
    nstride[0] /= sizeof(T0);
    if (STRENGTH > 1)
    {
        nxt1 = reinterpret_cast<const T0*>(nextp[1]);
        nstride[1] /= sizeof(T0);
    }
    if (STRENGTH > 2)
    {
        prv2 = reinterpret_cast<const T0*>(prevp[2]);
        nxt2 = reinterpret_cast<const T0*>(nextp[2]);
        pstride[2] /= sizeof(T0);
        nstride[2] /= sizeof(T0);
    }

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const T1 curx = static_cast<T1>(cur0[x]);
            T1 d1 = prv1[x] - curx;
            T1 d2 = 0;
            if (d1 < 0)
            {
                d2 = -d1;
                d1 = 0;
            }
            if (STRENGTH > 1)
            {
                update_diff(static_cast<T1>(nxt1[x]), curx, d1, d2);
            }
            if (STRENGTH > 2)
            {
                update_diff(static_cast<T1>(prv2[x]), curx, d1, d2);
                update_diff(static_cast<T1>(nxt2[x]), curx, d1, d2);
            }
            T1 prvx = static_cast<T1>(prv0[x]);
            T1 nxtx = static_cast<T1>(nxt0[x]);
            T1 avg = get_avg(prvx, nxtx, curx);
            T1 ul = max(min(prvx, nxtx) - d1, curx);
            T1 ll = min(max(prvx, nxtx) + d2, curx);
            dst0[x] = static_cast<T0>(clamp(avg, ll, ul));
        }
        dst0 += dstride;
        cur0 += cstride;
        prv0 += pstride[0];
        prv1 += pstride[1];
        nxt0 += nstride[0];
        if (STRENGTH > 1)
        {
            nxt1 += nstride[1];
        }
        if (STRENGTH > 2)
        {
            prv2 += pstride[2];
            nxt2 += nstride[2];
        }
    }
}

ReduceFlicker::ReduceFlicker(PClip c, int s, bool aggressive, bool grey, int opt, bool ra, bool luma, IScriptEnvironment* env) :
    GenericVideoFilter(c), strength(s), _grey(grey), opt_(opt), raccess(ra), _luma(luma)
{
    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; }

    int planecount = min(vi.NumComponents(), 3);
    for (int i = 0; i < planecount; ++i)
    {
        if (vi.IsRGB())
            processPlane[i] = true;
        else if (i == 0)
            processPlane[i] = _luma;
        else
            processPlane[i] = !_grey;
    }

    avx2 = ((!!(env->GetCPUFlags() & CPUF_AVX2) && opt_ < 0) || opt_ == 2);
    sse2 = ((!!(env->GetCPUFlags() & CPUF_SSE2) && opt_ < 0) || opt_ == 1);

    align = avx2 ? 32 : 16;

    if (avx2 && !aggressive)
        switch (vi.ComponentSize())
        {
            case 1:
                switch (strength)
                {
                    case 1: process = proc_avx2<uint8_t, 1>; break;
                    case 2: process = proc_avx2<uint8_t, 2>; break;
                    default: process = proc_avx2<uint8_t, 3>; break;
                }
                break;
            case 2:
                switch (strength)
                {
                    case 1: process = proc_avx2<uint16_t, 1>; break;
                    case 2: process = proc_avx2<uint16_t, 2>; break;
                    default: process = proc_avx2<uint16_t, 3>; break;
                }
                break;
            default:
                switch (strength)
                {
                    case 1: process = proc_avx2<float, 1>; break;
                    case 2: process = proc_avx2<float, 2>; break;
                    default: process = proc_avx2<float, 3>; break;
                }
                break;
        }
    else if (avx2 && aggressive)
        switch (vi.ComponentSize())
        {
            case 1:
                switch (strength)
                {
                    case 1: process = proc_a_avx2<uint8_t, 1>; break;
                    case 2: process = proc_a_avx2<uint8_t, 2>; break;
                    default: process = proc_a_avx2<uint8_t, 3>; break;
                }
                break;
            case 2:
                switch (strength)
                {
                    case 1: process = proc_a_avx2<uint16_t, 1>; break;
                    case 2: process = proc_a_avx2<uint16_t, 2>; break;
                    default: process = proc_a_avx2<uint16_t, 3>; break;
                }
                break;
            default:
                switch (strength)
                {
                    case 1: process = proc_a_avx2<float, 1>; break;
                    case 2: process = proc_a_avx2<float, 2>; break;
                    default: process = proc_a_avx2<float, 3>; break;
                }
                break;
        }
    else if (sse2 && !aggressive)
        switch (vi.ComponentSize())
        {
            case 1:
                switch (strength)
                {
                    case 1: process = proc_sse2<uint8_t, 1>; break;
                    case 2: process = proc_sse2<uint8_t, 2>; break;
                    default: process = proc_sse2<uint8_t, 3>; break;
                }
                break;
            case 2:
                switch (strength)
                {
                    case 1: process = proc_sse2<uint16_t, 1>; break;
                    case 2: process = proc_sse2<uint16_t, 2>; break;
                    default: process = proc_sse2<uint16_t, 3>; break;
                }
                break;
            default:
                switch (strength)
                {
                    case 1: process = proc_sse2<float, 1>; break;
                    case 2: process = proc_sse2<float, 2>; break;
                    default: process = proc_sse2<float, 3>; break;
                }
                break;
        }
    else if (sse2 && aggressive)
        switch (vi.ComponentSize())
        {
            case 1:
                switch (strength)
                {
                    case 1: process = proc_a_sse2<uint8_t, 1>; break;
                    case 2: process = proc_a_sse2<uint8_t, 2>; break;
                    default: process = proc_a_sse2<uint8_t, 3>; break;
                }
                break;
            case 2:
                switch (strength)
                {
                    case 1: process = proc_a_sse2<uint16_t, 1>; break;
                    case 2: process = proc_a_sse2<uint16_t, 2>; break;
                    default: process = proc_a_sse2<uint16_t, 3>; break;
                }
                break;
            default:
                switch (strength)
                {
                    case 1: process = proc_a_sse2<float, 1>; break;
                    case 2: process = proc_a_sse2<float, 2>; break;
                    default: process = proc_a_sse2<float, 3>; break;
                }
                break;
        }
    else if (!aggressive)
        switch (vi.ComponentSize())
        {
            case 1:
                switch (strength)
                {
                    case 1: process = proc_c<uint8_t, 1>; break;
                    case 2: process = proc_c<uint8_t, 2>; break;
                    default: process = proc_c<uint8_t, 3>; break;
                }
                break;
            case 2:
                switch (strength)
                {
                    case 1: process = proc_c<uint16_t, 1>; break;
                    case 2: process = proc_c<uint16_t, 2>; break;
                    default: process = proc_c<uint16_t, 3>; break;
                }
                break;
            default:
                switch (strength)
                {
                    case 1: process = proc_c<float, 1>; break;
                    case 2: process = proc_c<float, 2>; break;
                    default: process = proc_c<float, 3>; break;
                }
                break;
        }
    else
        switch (vi.ComponentSize())
        {
            case 1:
                switch (strength)
                {
                    case 1: process = proc_a_c<uint8_t, 1>; break;
                    case 2: process = proc_a_c<uint8_t, 2>; break;
                    default: process = proc_a_c<uint8_t, 3>; break;
                }
                break;
            case 2:
                switch (strength)
                {
                    case 1: process = proc_a_c<uint16_t, 1>; break;
                    case 2: process = proc_a_c<uint16_t, 2>; break;
                    default: process = proc_a_c<uint16_t, 3>; break;
                }
                break;
            default:
                switch (strength)
                {
                    case 1: process = proc_a_c<float, 1>; break;
                    case 2: process = proc_a_c<float, 2>; break;
                    default: process = proc_a_c<float, 3>; break;
                }
                break;
        }
}

PVideoFrame __stdcall ReduceFlicker::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame curr, prev[3], next[3];
    const int nf = vi.num_frames - 1;

    if (raccess)
        switch (strength)
        {
            case 3:
                next[2] = child->GetFrame(min(n + 3, nf), env);
                next[1] = child->GetFrame(min(n + 2, nf), env);
                next[0] = child->GetFrame(min(n + 1, nf), env);
                curr = child->GetFrame(n, env);
                prev[0] = child->GetFrame(max(n - 1, 0), env);
                prev[1] = child->GetFrame(max(n - 2, 0), env);
                prev[2] = child->GetFrame(max(n - 3, 0), env);
                break;
            case 2:
                next[1] = child->GetFrame(min(n + 2, nf), env);
                next[0] = child->GetFrame(min(n + 1, nf), env);
                curr = child->GetFrame(n, env);
                prev[0] = child->GetFrame(max(n - 1, 0), env);
                prev[1] = child->GetFrame(max(n - 2, 0), env);
                break;
            default:
                next[0] = child->GetFrame(min(n + 1, nf), env);
                curr = child->GetFrame(n, env);
                prev[1] = child->GetFrame(max(n - 2, 0), env);
                prev[0] = child->GetFrame(max(n - 1, 0), env);
                break;
        }
    else
        switch (strength)
        {
            case 3:
                prev[2] = child->GetFrame(max(n - 3, 0), env);
                prev[1] = child->GetFrame(max(n - 2, 0), env);
                prev[0] = child->GetFrame(max(n - 1, 0), env);
                curr = child->GetFrame(n, env);
                next[0] = child->GetFrame(min(n + 1, nf), env);
                next[1] = child->GetFrame(min(n + 2, nf), env);
                next[2] = child->GetFrame(min(n + 3, nf), env);
                break;
            case 2:
                prev[1] = child->GetFrame(max(n - 2, 0), env);
                prev[0] = child->GetFrame(max(n - 1, 0), env);
                curr = child->GetFrame(n, env);
                next[0] = child->GetFrame(min(n + 1, nf), env);
                next[1] = child->GetFrame(min(n + 2, nf), env);
                break;
            default:
                prev[1] = child->GetFrame(max(n - 2, 0), env);
                prev[0] = child->GetFrame(max(n - 1, 0), env);
                curr = child->GetFrame(n, env);
                next[0] = child->GetFrame(min(n + 1, nf), env);
                break;
        }

    PVideoFrame dst = has_at_least_v8 ? env->NewVideoFrameP(vi, &curr, align) : env->NewVideoFrame(vi, align);

    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    const int* current_planes = !vi.IsRGB() ? planes_y : planes_r;
    int planecount = min(vi.NumComponents(), 3);
    for (int i = 0; i < planecount; ++i)
    {
        const int plane = current_planes[i];

        if (processPlane[i])
        {
            int width = curr->GetRowSize(plane) / vi.ComponentSize();
            int height = curr->GetHeight(plane);
            int cpitch = curr->GetPitch(plane);
            int dpitch = dst->GetPitch(plane);
            const uint8_t* currp = curr->GetReadPtr(plane);
            uint8_t* dstp = dst->GetWritePtr(plane);

            const uint8_t* prevp[3], * nextp[3];
            int ppitch[3], npitch[3];

            switch (strength)
            {
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

            process(dstp, currp, prevp, nextp, dpitch, cpitch, ppitch, npitch, width, height);
        }
    }

    return dst;
}

AVSValue __cdecl Create_ReduceFlicker(AVSValue args, void*, IScriptEnvironment* env)
{
    PClip clip = args[0].AsClip();
    const VideoInfo& vi = clip->GetVideoInfo();

    if (!vi.IsPlanar())
        env->ThrowError("ReduceFlicker: input clip must be in planar format.");

    int strength = args[1].AsInt(2);
    if (strength < 1 || strength > 3)
        env->ThrowError("ReduceFlicker: strength must be set to 1, 2 or 3.");

    int opt = args[4].AsInt(-1);
    if (opt < -1 || opt > 2)
        env->ThrowError("ReduceFlicker: opt must be between -1..2.");
    if (!(env->GetCPUFlags() & CPUF_AVX2) && opt == 2)
        env->ThrowError("ReduceFlicker: opt=2 requires AVX2.");
    if (!(env->GetCPUFlags() & CPUF_SSE2) && opt == 1)
        env->ThrowError("ReduceFlicker: opt=1 requires SSE2.");

    return new ReduceFlicker(
        clip,
        strength,
        args[2].AsBool(false),
        args[3].AsBool(false),
        opt,
        args[5].AsBool(true),
        args[6].AsBool(true),
        env);
}

const AVS_Linkage* AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    env->AddFunction("ReduceFlicker", "c[strength]i[aggressive]b[grey]b[opt]i[raccess]b[luma]b", Create_ReduceFlicker, 0);

    return "ReduceFlicker for avs2.6/avs+.";
}
