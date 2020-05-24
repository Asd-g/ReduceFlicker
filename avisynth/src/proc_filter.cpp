/*
proc_filter.cpp

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

#ifndef _PROC_FILTER_H
#define _PROC_FILTER_H

#include <cstdint>
#include <algorithm>
#include <map>
#include <tuple>

#include "ReduceFlicker.h"
#include "simd.h"

template <typename T>
static F_INLINE T absdiff(const T x, const T y)
{
    return x < y ? y - x : x - y;
}

template <typename T>
static F_INLINE T clamp(T x, T minimum, T maximum)
{
    return std::min(std::max(x, minimum), maximum);
}

template <typename T>
static F_INLINE T get_avg(T a, T b, T x)
{
    int t = std::max((a + b + 1) / 2 - 1, 0);
    return static_cast<T>((t + x + 1) / 2);
}

template <>
F_INLINE float get_avg(float a, float b, float x)
{
    return (a + b + x + x) * 0.25f;
}


template <typename T0, typename T1, int STRENGTH>
static void __stdcall
proc_c(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp,
    const uint8_t** nextp, int dstride, int cstride, int* pstride,
    int* nstride, int width, int height)
    noexcept
{
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
    if (STRENGTH > 1) {
        nxt1 = reinterpret_cast<const T0*>(nextp[1]);
        nstride[1] /= sizeof(T0);
    }
    if (STRENGTH > 2) {
        prv2 = reinterpret_cast<const T0*>(prevp[2]);
        nxt2 = reinterpret_cast<const T0*>(nextp[2]);
        pstride[2] /= sizeof(T0);
        nstride[2] /= sizeof(T0);
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const T1 curx = static_cast<T1>(cur0[x]);
            T1 d = absdiff(curx, static_cast<T1>(prv1[x]));
            if (STRENGTH > 1) {
                d = std::min(d, absdiff(curx, static_cast<T1>(nxt1[x])));
            }
            if (STRENGTH > 2) {
                d = std::min(d, absdiff(curx, static_cast<T1>(prv2[x])));
                d = std::min(d, absdiff(curx, static_cast<T1>(nxt2[x])));
            }
            T1 prvx = static_cast<T1>(prv0[x]);
            T1 nxtx = static_cast<T1>(nxt0[x]);
            T1 avg = get_avg(prvx, nxtx, curx);
            T1 ul = std::max(std::min(prvx, nxtx) - d, curx);
            T1 ll = std::min(std::max(prvx, nxtx) + d, curx);
            dst0[x] = static_cast<T0>(clamp(avg, ll, ul));
        }
        dst0 += dstride;
        cur0 += cstride;
        prv0 += pstride[0];
        prv1 += pstride[1];
        nxt0 += nstride[0];
        if (STRENGTH > 1) {
            nxt1 += nstride[1];
        }
        if (STRENGTH > 2) {
            prv2 += pstride[2];
            nxt2 += nstride[2];
        }
    }
}


template <typename T>
static F_INLINE void update_diff(T x, T y, T& d1, T& d2)
{
    T d = x - y;
    if (d >= 0) {
        d2 = 0;
        d1 = std::min(d, d1);
    }
    else {
        d1 = 0;
        d2 = std::min(-d, d2);
    }
}


template <typename T0, typename T1, int STRENGTH>
static void __stdcall
proc_a_c(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp,
    const uint8_t** nextp, int dstride, int cstride, int* pstride,
    int* nstride, int width, int height)
    noexcept
{
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
    if (STRENGTH > 1) {
        nxt1 = reinterpret_cast<const T0*>(nextp[1]);
        nstride[1] /= sizeof(T0);
    }
    if (STRENGTH > 2) {
        prv2 = reinterpret_cast<const T0*>(prevp[2]);
        nxt2 = reinterpret_cast<const T0*>(nextp[2]);
        pstride[2] /= sizeof(T0);
        nstride[2] /= sizeof(T0);
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const T1 curx = static_cast<T1>(cur0[x]);
            T1 d1 = prv1[x] - curx;
            T1 d2 = 0;
            if (d1 < 0) {
                d2 = -d1;
                d1 = 0;
            }
            if (STRENGTH > 1) {
                update_diff(static_cast<T1>(nxt1[x]), curx, d1, d2);
            }
            if (STRENGTH > 2) {
                update_diff(static_cast<T1>(prv2[x]), curx, d1, d2);
                update_diff(static_cast<T1>(nxt2[x]), curx, d1, d2);
            }
            T1 prvx = static_cast<T1>(prv0[x]);
            T1 nxtx = static_cast<T1>(nxt0[x]);
            T1 avg = get_avg(prvx, nxtx, curx);
            T1 ul = std::max(std::min(prvx, nxtx) - d1, curx);
            T1 ll = std::min(std::max(prvx, nxtx) + d2, curx);
            dst0[x] = static_cast<T0>(clamp(avg, ll, ul));
        }
        dst0 += dstride;
        cur0 += cstride;
        prv0 += pstride[0];
        prv1 += pstride[1];
        nxt0 += nstride[0];
        if (STRENGTH > 1) {
            nxt1 += nstride[1];
        }
        if (STRENGTH > 2) {
            prv2 += pstride[2];
            nxt2 += nstride[2];
        }
    }
}


/****************************** SIMD version *********************/

#if defined(__SSE2__)

template <typename T, typename V, int STRENGTH, arch_t ARCH>
static void __stdcall
proc_simd(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp,
    const uint8_t** nextp, int dstride, int cstride, int* pstride,
    int* nstride, int width, int height)
    noexcept
{
    const uint8_t* prv0, * prv1, * prv2, * nxt0, * nxt1, * nxt2;
    prv0 = prevp[0];
    prv1 = prevp[1];
    nxt0 = nextp[0];
    if (STRENGTH > 1) {
        nxt1 = nextp[1];
    }
    if (STRENGTH > 2) {
        prv2 = prevp[2];
        nxt2 = nextp[2];
    }

    width *= sizeof(T);

    V q = set1<T, V>();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; x += sizeof(V)) {
            const V curx = load<V>(currp + x);
            V d = abs_diff<T, V>(curx, load<V>(prv1 + x));
            if (STRENGTH > 1) {
                d = min<T, ARCH>(d, abs_diff<T, V>(curx, load<V>(nxt1 + x)));
            }
            if (STRENGTH > 2) {
                d = min<T, ARCH>(d, abs_diff<T, V>(curx, load<V>(prv2 + x)));
                d = min<T, ARCH>(d, abs_diff<T, V>(curx, load<V>(nxt2 + x)));
            }
            const V pr0 = load<V>(prv0 + x);
            const V nx0 = load<V>(nxt0 + x);
            const V ul = max<T, ARCH>(sub<T>(min<T, ARCH>(pr0, nx0), d), curx);
            const V ll = min<T, ARCH>(add<T>(max<T, ARCH>(pr0, nx0), d), curx);
            const V avg = get_avg<T, V>(pr0, nx0, curx, q);
            stream(dstp + x, clamp<T, V, ARCH>(avg, ll, ul));
        }
        prv0 += pstride[0];
        prv1 += pstride[1];
        nxt0 += nstride[0];
        currp += cstride;
        dstp += dstride;
        if (STRENGTH > 1) {
            nxt1 += nstride[1];
        }
        if (STRENGTH > 2) {
            prv2 += pstride[2];
            nxt2 += nstride[2];
        }
    }
}


template <typename T, typename V, arch_t ARCH>
static F_INLINE void
update_diff(const V& x, const V& y, V& d1, V& d2, const V& zero)
{
    const V maxxy = max<T, ARCH>(x, y);
    const V mask = cmpeq<T>(x, maxxy);
    const V d = sub<T>(maxxy, min<T, ARCH>(x, y));
    d1 = blendv<ARCH>(zero, min<T, ARCH>(d, d1), mask);
    d2 = blendv<ARCH>(min<T, ARCH>(d, d2), zero, mask);
}


template <typename T, typename V, int STRENGTH, arch_t ARCH>
static void __stdcall
proc_a_simd(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp,
    const uint8_t** nextp, int dstride, int cstride, int* pstride,
    int* nstride, int width, int height)
    noexcept
{
    const uint8_t* prv0, * prv1, * prv2, * nxt0, * nxt1, * nxt2;
    prv0 = prevp[0];
    prv1 = prevp[1];
    nxt0 = nextp[0];
    if (STRENGTH > 1) {
        nxt1 = nextp[1];
    }
    if (STRENGTH > 2) {
        prv2 = prevp[2];
        nxt2 = nextp[2];
    }

    width *= sizeof(T);

    V q = set1<T, V>();
    V zero = setzero<V>();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; x += sizeof(V)) {
            const V curx = load<V>(currp + x);
            V t0 = load<V>(prv1 + x);
            V t1 = max<T, ARCH>(t0, curx);
            V t2 = cmpeq<T>(t0, t1);
            t0 = sub<T>(t1, min<T, ARCH>(t0, curx));
            V d1 = and_reg(t2, t0);
            V d2 = andnot_reg(t2, t0);
            if (STRENGTH > 1) {
                update_diff<T, V, ARCH>(load<V>(nxt1 + x), curx, d1, d2, zero);
            }
            if (STRENGTH > 2) {
                update_diff<T, V, ARCH>(load<V>(prv2 + x), curx, d1, d2, zero);
                update_diff<T, V, ARCH>(load<V>(nxt2 + x), curx, d1, d2, zero);
            }
            const V pr0 = load<V>(prv0 + x);
            const V nx0 = load<V>(nxt0 + x);
            const V ul = max<T, ARCH>(sub<T>(min<T, ARCH>(pr0, nx0), d1), curx);
            const V ll = min<T, ARCH>(add<T>(max<T, ARCH>(pr0, nx0), d2), curx);
            const V avg = get_avg<T, V>(pr0, nx0, curx, q);
            stream(dstp + x, clamp<T, V, ARCH>(avg, ll, ul));
        }
        prv0 += pstride[0];
        prv1 += pstride[1];
        nxt0 += nstride[0];
        currp += cstride;
        dstp += dstride;
        if (STRENGTH > 1) {
            nxt1 += nstride[1];
        }
        if (STRENGTH > 2) {
            prv2 += pstride[2];
            nxt2 += nstride[2];
        }
    }

}

#endif // __SSE2__

proc_filter_t
get_main_proc(arch_t arch, int strength, bool aggressive, int bits_per_sample)
{
    using std::make_tuple;

    std::map<std::tuple<arch_t, int, bool, int>, proc_filter_t> table;

    table[make_tuple(arch_t::NO_SIMD, 1, false, 8)] = proc_c<uint8_t, int, 1>;
    table[make_tuple(arch_t::NO_SIMD, 1, false, 16)] = proc_c<uint16_t, int, 1>;
    table[make_tuple(arch_t::NO_SIMD, 1, false, 32)] = proc_c<float, float, 1>;

    table[make_tuple(arch_t::NO_SIMD, 2, false, 8)] = proc_c<uint8_t, int, 2>;
    table[make_tuple(arch_t::NO_SIMD, 2, false, 16)] = proc_c<uint16_t, int, 2>;
    table[make_tuple(arch_t::NO_SIMD, 2, false, 32)] = proc_c<float, float, 2>;

    table[make_tuple(arch_t::NO_SIMD, 3, false, 8)] = proc_c<uint8_t, int, 3>;
    table[make_tuple(arch_t::NO_SIMD, 3, false, 16)] = proc_c<uint16_t, int, 3>;
    table[make_tuple(arch_t::NO_SIMD, 3, false, 32)] = proc_c<float, float, 3>;

    table[make_tuple(arch_t::NO_SIMD, 1, true, 8)] = proc_a_c<uint8_t, int, 1>;
    table[make_tuple(arch_t::NO_SIMD, 1, true, 16)] = proc_a_c<uint16_t, int, 1>;
    table[make_tuple(arch_t::NO_SIMD, 1, true, 32)] = proc_a_c<float, float, 1>;

    table[make_tuple(arch_t::NO_SIMD, 2, true, 8)] = proc_a_c<uint8_t, int, 2>;
    table[make_tuple(arch_t::NO_SIMD, 2, true, 16)] = proc_a_c<uint16_t, int, 2>;
    table[make_tuple(arch_t::NO_SIMD, 2, true, 32)] = proc_a_c<float, float, 2>;

    table[make_tuple(arch_t::NO_SIMD, 3, true, 8)] = proc_a_c<uint8_t, int, 3>;
    table[make_tuple(arch_t::NO_SIMD, 3, true, 16)] = proc_a_c<uint16_t, int, 3>;
    table[make_tuple(arch_t::NO_SIMD, 3, true, 32)] = proc_a_c<float, float, 3>;

#if defined(__SSE2__)
    table[make_tuple(arch_t::USE_SSE2, 1, false, 8)] = proc_simd<uint8_t, __m128i, 1, arch_t::USE_SSE2>;
    table[make_tuple(arch_t::USE_SSE2, 1, false, 16)] = proc_simd<uint16_t, __m128i, 1, arch_t::USE_SSE2>;
    table[make_tuple(arch_t::USE_SSE2, 1, false, 32)] = proc_simd<float, __m128, 1, arch_t::USE_SSE2>;

    table[make_tuple(arch_t::USE_SSE2, 2, false, 8)] = proc_simd<uint8_t, __m128i, 2, arch_t::USE_SSE2>;
    table[make_tuple(arch_t::USE_SSE2, 2, false, 16)] = proc_simd<uint16_t, __m128i, 2, arch_t::USE_SSE2>;
    table[make_tuple(arch_t::USE_SSE2, 2, false, 32)] = proc_simd<float, __m128, 2, arch_t::USE_SSE2>;

    table[make_tuple(arch_t::USE_SSE2, 3, false, 8)] = proc_simd<uint8_t, __m128i, 3, arch_t::USE_SSE2>;
    table[make_tuple(arch_t::USE_SSE2, 3, false, 16)] = proc_simd<uint16_t, __m128i, 3, arch_t::USE_SSE2>;
    table[make_tuple(arch_t::USE_SSE2, 3, false, 32)] = proc_simd<float, __m128, 3, arch_t::USE_SSE2>;

    table[make_tuple(arch_t::USE_SSE2, 1, true, 8)] = proc_a_simd<uint8_t, __m128i, 1, arch_t::USE_SSE2>;
    table[make_tuple(arch_t::USE_SSE2, 1, true, 16)] = proc_a_simd<uint16_t, __m128i, 1, arch_t::USE_SSE2>;
    table[make_tuple(arch_t::USE_SSE2, 1, true, 32)] = proc_a_simd<float, __m128, 1, arch_t::USE_SSE2>;

    table[make_tuple(arch_t::USE_SSE2, 2, true, 8)] = proc_a_simd<uint8_t, __m128i, 2, arch_t::USE_SSE2>;
    table[make_tuple(arch_t::USE_SSE2, 2, true, 16)] = proc_a_simd<uint16_t, __m128i, 2, arch_t::USE_SSE2>;
    table[make_tuple(arch_t::USE_SSE2, 2, true, 32)] = proc_a_simd<float, __m128, 2, arch_t::USE_SSE2>;

    table[make_tuple(arch_t::USE_SSE2, 3, true, 8)] = proc_a_simd<uint8_t, __m128i, 3, arch_t::USE_SSE2>;
    table[make_tuple(arch_t::USE_SSE2, 3, true, 16)] = proc_a_simd<uint16_t, __m128i, 3, arch_t::USE_SSE2>;
    table[make_tuple(arch_t::USE_SSE2, 3, true, 32)] = proc_a_simd<float, __m128, 3, arch_t::USE_SSE2>;

#if defined(__SSE4_1__)
    table[make_tuple(arch_t::USE_SSE41, 1, false, 8)] = proc_simd<uint8_t, __m128i, 1, arch_t::USE_SSE2>;
    table[make_tuple(arch_t::USE_SSE41, 1, false, 16)] = proc_simd<uint16_t, __m128i, 1, arch_t::USE_SSE41>;
    table[make_tuple(arch_t::USE_SSE41, 1, false, 32)] = proc_simd<float, __m128, 1, arch_t::USE_SSE2>;

    table[make_tuple(arch_t::USE_SSE41, 2, false, 8)] = proc_simd<uint8_t, __m128i, 2, arch_t::USE_SSE2>;
    table[make_tuple(arch_t::USE_SSE41, 2, false, 16)] = proc_simd<uint16_t, __m128i, 2, arch_t::USE_SSE41>;
    table[make_tuple(arch_t::USE_SSE41, 2, false, 32)] = proc_simd<float, __m128, 2, arch_t::USE_SSE2>;

    table[make_tuple(arch_t::USE_SSE41, 3, false, 8)] = proc_simd<uint8_t, __m128i, 3, arch_t::USE_SSE2>;
    table[make_tuple(arch_t::USE_SSE41, 3, false, 16)] = proc_simd<uint16_t, __m128i, 3, arch_t::USE_SSE41>;
    table[make_tuple(arch_t::USE_SSE41, 3, false, 32)] = proc_simd<float, __m128, 3, arch_t::USE_SSE2>;

    table[make_tuple(arch_t::USE_SSE41, 1, true, 8)] = proc_a_simd<uint8_t, __m128i, 1, arch_t::USE_SSE41>;
    table[make_tuple(arch_t::USE_SSE41, 1, true, 16)] = proc_a_simd<uint16_t, __m128i, 1, arch_t::USE_SSE41>;
    table[make_tuple(arch_t::USE_SSE41, 1, true, 32)] = proc_a_simd<float, __m128, 1, arch_t::USE_SSE41>;

    table[make_tuple(arch_t::USE_SSE41, 2, true, 8)] = proc_a_simd<uint8_t, __m128i, 2, arch_t::USE_SSE41>;
    table[make_tuple(arch_t::USE_SSE41, 2, true, 16)] = proc_a_simd<uint16_t, __m128i, 2, arch_t::USE_SSE41>;
    table[make_tuple(arch_t::USE_SSE41, 2, true, 32)] = proc_a_simd<float, __m128, 2, arch_t::USE_SSE41>;

    table[make_tuple(arch_t::USE_SSE41, 3, true, 8)] = proc_a_simd<uint8_t, __m128i, 3, arch_t::USE_SSE41>;
    table[make_tuple(arch_t::USE_SSE41, 3, true, 16)] = proc_a_simd<uint16_t, __m128i, 3, arch_t::USE_SSE41>;
    table[make_tuple(arch_t::USE_SSE41, 3, true, 32)] = proc_a_simd<float, __m128, 3, arch_t::USE_SSE41>;

#if defined(__AVX2__)
    table[make_tuple(arch_t::USE_AVX2, 1, false, 8)] = proc_simd<uint8_t, __m256i, 1, arch_t::USE_AVX2>;
    table[make_tuple(arch_t::USE_AVX2, 1, false, 16)] = proc_simd<uint16_t, __m256i, 1, arch_t::USE_AVX2>;
    table[make_tuple(arch_t::USE_AVX2, 1, false, 32)] = proc_simd<float, __m128, 1, arch_t::USE_AVX2>;

    table[make_tuple(arch_t::USE_AVX2, 2, false, 8)] = proc_simd<uint8_t, __m256i, 2, arch_t::USE_AVX2>;
    table[make_tuple(arch_t::USE_AVX2, 2, false, 16)] = proc_simd<uint16_t, __m256i, 2, arch_t::USE_AVX2>;
    table[make_tuple(arch_t::USE_AVX2, 2, false, 32)] = proc_simd<float, __m128, 2, arch_t::USE_AVX2>;

    table[make_tuple(arch_t::USE_AVX2, 3, false, 8)] = proc_simd<uint8_t, __m256i, 3, arch_t::USE_AVX2>;
    table[make_tuple(arch_t::USE_AVX2, 3, false, 16)] = proc_simd<uint16_t, __m256i, 3, arch_t::USE_AVX2>;
    table[make_tuple(arch_t::USE_AVX2, 3, false, 32)] = proc_simd<float, __m256, 3, arch_t::USE_AVX2>;

    table[make_tuple(arch_t::USE_AVX2, 1, true, 8)] = proc_a_simd<uint8_t, __m256i, 1, arch_t::USE_AVX2>;
    table[make_tuple(arch_t::USE_AVX2, 1, true, 16)] = proc_a_simd<uint16_t, __m256i, 1, arch_t::USE_AVX2>;
    table[make_tuple(arch_t::USE_AVX2, 1, true, 32)] = proc_a_simd<float, __m256, 1, arch_t::USE_AVX2>;

    table[make_tuple(arch_t::USE_AVX2, 2, true, 8)] = proc_a_simd<uint8_t, __m256i, 2, arch_t::USE_AVX2>;
    table[make_tuple(arch_t::USE_AVX2, 2, true, 16)] = proc_a_simd<uint16_t, __m256i, 2, arch_t::USE_AVX2>;
    table[make_tuple(arch_t::USE_AVX2, 2, true, 32)] = proc_a_simd<float, __m256, 2, arch_t::USE_AVX2>;

    table[make_tuple(arch_t::USE_AVX2, 3, true, 8)] = proc_a_simd<uint8_t, __m256i, 3, arch_t::USE_AVX2>;
    table[make_tuple(arch_t::USE_AVX2, 3, true, 16)] = proc_a_simd<uint16_t, __m256i, 3, arch_t::USE_AVX2>;
    table[make_tuple(arch_t::USE_AVX2, 3, true, 32)] = proc_a_simd<float, __m256, 3, arch_t::USE_AVX2>;
#endif
#endif
#endif

    return table[make_tuple(arch, strength, aggressive, bits_per_sample)];

}

#endif