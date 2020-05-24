/*
simd.h

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

#ifndef _SIMD_H
#define _SIMD_H


#include <immintrin.h>

#include "ReduceFlicker.h"

#if defined(__SSE2__)

#define F_INLINE __forceinline

/********************* LOAD ****************************************/
template <typename V> static F_INLINE V load(const uint8_t* p);

template <>
F_INLINE __m128i load<__m128i>(const uint8_t* p)
{
    return _mm_load_si128(reinterpret_cast<const __m128i*>(p));
}
template <>
F_INLINE __m128 load(const uint8_t* p)
{
    return _mm_load_ps(reinterpret_cast<const float*>(p));
}
#if defined(__AVX2__)
template <>
F_INLINE __m256i load(const uint8_t* p)
{
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
}
template <>
F_INLINE __m256 load(const uint8_t* p)
{
    return _mm256_load_ps(reinterpret_cast<const float*>(p));
}
#endif

/********************* STORE *************************************/
static F_INLINE void stream(uint8_t* p, const __m128i& x)
{
    _mm_stream_si128(reinterpret_cast<__m128i*>(p), x);
}

static F_INLINE void stream(uint8_t* p, const __m128& x)
{
    _mm_stream_ps(reinterpret_cast<float*>(p), x);
}

#if defined(__AVX2__)
static F_INLINE void stream(uint8_t* p, const __m256i& x)
{
    _mm256_stream_si256(reinterpret_cast<__m256i*>(p), x);
}

static F_INLINE void stream(uint8_t* p, const __m256& x)
{
    _mm256_stream_ps(reinterpret_cast<float*>(p), x);
}
#endif

/************************ SETZERO *********************************/
template <typename V> static F_INLINE V setzero();

template <>
F_INLINE __m128i setzero<__m128i>()
{
    return _mm_setzero_si128();
}
template <>
F_INLINE __m128 setzero<__m128>()
{
    return _mm_setzero_ps();
}

#if defined(__AVX2__)
template <>
F_INLINE __m256i setzero<__m256i>()
{
    return _mm256_setzero_si256();
}
template <>
F_INLINE __m256 setzero<__m256>()
{
    return _mm256_setzero_ps();
}
#endif


/*********************** SET1 *************************************/
template <typename T, typename V> static F_INLINE V set1();

template <>
F_INLINE __m128i set1<uint8_t>()
{
    __m128i zero = _mm_setzero_si128();
    return _mm_sub_epi8(zero, _mm_cmpeq_epi32(zero, zero));
}
template <>
F_INLINE __m128i set1<int16_t>()
{
    __m128i zero = _mm_setzero_si128();
    return _mm_sub_epi16(zero, _mm_cmpeq_epi32(zero, zero));
}
template <>
F_INLINE __m128i set1<uint16_t>()
{
    return set1<int16_t, __m128i>();
}
template <>
F_INLINE __m128 set1<float>()
{
    return _mm_set1_ps(0.25f);
}

#if defined(__AVX2__)
template <>
F_INLINE __m256i set1<uint8_t>()
{
    __m256i zero = _mm256_setzero_si256();
    return _mm256_sub_epi8(zero, _mm256_cmpeq_epi32(zero, zero));
}
template <>
F_INLINE __m256i set1<uint16_t>()
{
    __m256i zero = _mm256_setzero_si256();
    return _mm256_sub_epi16(zero, _mm256_cmpeq_epi16(zero, zero));
}
template <>
F_INLINE __m256 set1<float>()
{
    return _mm256_set1_ps(0.25f);
}
#endif

/********************* BIT OR *************************************/
static F_INLINE __m128 or_reg(const __m128& x, const __m128& y)
{
    return _mm_or_ps(x, y);
}
static F_INLINE __m128i or_reg(const __m128i& x, const __m128i& y)
{
    return _mm_or_si128(x, y);
}

#if defined(__AVX2__)
static F_INLINE __m256 or_reg(const __m256& x, const __m256& y)
{
    return _mm256_or_ps(x, y);
}
static F_INLINE __m256i or_reg(const __m256i& x, const __m256i& y)
{
    return _mm256_or_si256(x, y);
}
#endif

/********************* BIT AND *************************************/
static F_INLINE __m128 and_reg(const __m128& x, const __m128& y)
{
    return _mm_and_ps(x, y);
}
static F_INLINE __m128i and_reg(const __m128i& x, const __m128i& y)
{
    return _mm_and_si128(x, y);
}

#if defined(__AVX2__)
static F_INLINE __m256 and_reg(const __m256& x, const __m256& y)
{
    return _mm256_and_ps(x, y);
}
static F_INLINE __m256i and_reg(const __m256i& x, const __m256i& y)
{
    return _mm256_and_si256(x, y);
}
#endif

/********************* BIT XOR *************************************/
static F_INLINE __m128 xor_reg(const __m128& x, const __m128& y)
{
    return _mm_xor_ps(x, y);
}
static F_INLINE __m128i xor_reg(const __m128i& x, const __m128i& y)
{
    return _mm_xor_si128(x, y);
}

#if defined(__AVX2__)
static F_INLINE __m256 xor_reg(const __m256& x, const __m256& y)
{
    return _mm256_xor_ps(x, y);
}
static F_INLINE __m256i xor_reg(const __m256i& x, const __m256i& y)
{
    return _mm256_xor_si256(x, y);
}
#endif

/********************* BIT ANDNOT *********************************/
static F_INLINE __m128 andnot_reg(const __m128& x, const __m128& y)
{
    return _mm_andnot_ps(x, y);
}
static F_INLINE __m128i andnot_reg(const __m128i& x, const __m128i& y)
{
    return _mm_andnot_si128(x, y);
}

#if defined(__AVX2__)
static F_INLINE __m256 andnot_reg(const __m256& x, const __m256& y)
{
    return _mm256_andnot_ps(x, y);
}
static F_INLINE __m256i andnot_reg(const __m256i& x, const __m256i& y)
{
    return _mm256_andnot_si256(x, y);
}
#endif

/************************ COMPEQ *********************************/
template <typename T>
static F_INLINE __m128i cmpeq(const __m128i& x, const __m128i& y)
{
    return _mm_cmpeq_epi16(x, y);
}
template <>
F_INLINE __m128i cmpeq<uint8_t>(const __m128i& x, const __m128i& y)
{
    return _mm_cmpeq_epi8(x, y);
}
template <typename T>
F_INLINE __m128 cmpeq(const __m128& x, const __m128& y)
{
    return _mm_cmpeq_ps(x, y);
}

#if defined(__AVX2__)
template <typename T>
static F_INLINE __m256i cmpeq(const __m256i& x, const __m256i& y)
{
    return _mm256_cmpeq_epi16(x, y);
}
template <>
F_INLINE __m256i cmpeq<uint8_t>(const __m256i& x, const __m256i& y)
{
    return _mm256_cmpeq_epi8(x, y);
}
template <typename T>
static F_INLINE __m256 cmpeq(const __m256& x, const __m256& y)
{
    return _mm256_cmp_ps(x, y, _CMP_EQ_OQ);
}
#endif

/********************** SUB **************************************/
template <typename T>
__m128i sub(const __m128i& x, const __m128i& y)
{
    return _mm_subs_epu16(x, y);
}

template <>
F_INLINE __m128i sub<uint8_t>(const __m128i& x, const __m128i& y)
{
    return _mm_subs_epu8(x, y);
}

template <typename T>
static F_INLINE __m128 sub(const __m128& x, const __m128& y)
{
    return _mm_sub_ps(x, y);
}

#if defined(__AVX2__)
template <typename T>
static F_INLINE __m256i sub(const __m256i& x, const __m256i& y)
{
    return _mm256_subs_epu16(x, y);
}
template <>
F_INLINE __m256i sub<uint8_t>(const __m256i& x, const __m256i& y)
{
    return _mm256_subs_epu8(x, y);
}
template <typename T>
static F_INLINE __m256 sub(const __m256& x, const __m256& y)
{
    return _mm256_sub_ps(x, y);
}
#endif

/************************** ADD *************************************/
template <typename T>
static F_INLINE __m128i add(const __m128i& x, const __m128i& y)
{
    return _mm_adds_epu16(x, y);
}

template <>
F_INLINE __m128i add<uint8_t>(const __m128i& x, const __m128i& y)
{
    return _mm_adds_epu8(x, y);
}

template <typename T>
static F_INLINE __m128 add(const __m128& x, const __m128& y)
{
    return _mm_add_ps(x, y);
}

#if defined(__AVX2__)
template <typename T>
static F_INLINE __m256i add(const __m256i& x, const __m256i& y)
{
    return _mm256_adds_epu16(x, y);
}
template <>
F_INLINE __m256i add<uint8_t>(const __m256i& x, const __m256i& y)
{
    return _mm256_adds_epu8(x, y);
}
template <typename T>
static F_INLINE __m256 add(const __m256& x, const __m256& y)
{
    return _mm256_add_ps(x, y);
}
#endif

/************************ MAX ************************************/
template <typename T, arch_t ARCH>
static F_INLINE __m128 max(const __m128& x, const __m128& y)
{
    return _mm_max_ps(x, y);
}

template <typename T, arch_t ARCH>
static F_INLINE __m128i max(const __m128i& x, const __m128i& y)
{
    return _mm_adds_epu16(y, _mm_subs_epu16(x, y));
}

template <>
F_INLINE __m128i max<uint8_t, arch_t::USE_SSE2>(const __m128i& x, const __m128i& y)
{
    return _mm_max_epu8(x, y);
}
template <>
F_INLINE __m128i max<int16_t, arch_t::USE_SSE2>(const __m128i& x, const __m128i& y)
{
    return _mm_max_epi16(x, y);
}

#if defined(__SSE4_1__)
template <>
F_INLINE __m128i max<uint8_t, arch_t::USE_SSE41>(const __m128i& x, const __m128i& y)
{
    return _mm_max_epu8(x, y);
}
template <>
F_INLINE __m128i max<uint16_t, arch_t::USE_SSE41>(const __m128i& x, const __m128i& y)
{
    return _mm_max_epu16(x, y);
}

#if defined(__AVX2__)
template <typename T, arch_t ARCH>
static F_INLINE __m256 max(const __m256& x, const __m256& y)
{
    return _mm256_max_ps(x, y);
}

template <typename T, arch_t ARCH>
static F_INLINE __m256i max(const __m256i& x, const __m256i& y)
{
    return _mm256_max_epu16(x, y);
}
template <>
F_INLINE __m256i max<uint8_t, arch_t::USE_AVX2>(const __m256i& x, const __m256i& y)
{
    return _mm256_max_epu8(x, y);
}
#endif // __AVX2__
#endif // __SSE4_1__


/************************ MIN ************************************/
template <typename T, arch_t ARCH>
static F_INLINE __m128 min(const __m128& x, const __m128& y)
{
    return _mm_min_ps(x, y);
}

template <typename T, arch_t ARCH>
static F_INLINE __m128i min(const __m128i& x, const __m128i& y)
{
    return _mm_subs_epu16(x, _mm_subs_epu16(x, y));
}

template <>
F_INLINE __m128i min<uint8_t, arch_t::USE_SSE2>(const __m128i& x, const __m128i& y)
{
    return _mm_min_epu8(x, y);
}
template <>
F_INLINE __m128i min<int16_t, arch_t::USE_SSE2>(const __m128i& x, const __m128i& y)
{
    return _mm_min_epi16(x, y);
}

#if defined(__SSE4_1__)
template <>
F_INLINE __m128i min<uint8_t, arch_t::USE_SSE41>(const __m128i& x, const __m128i& y)
{
    return _mm_min_epu8(x, y);
}
template <>
F_INLINE __m128i min<uint16_t, arch_t::USE_SSE41>(const __m128i& x, const __m128i& y)
{
    return _mm_min_epu16(x, y);
}

#if defined(__AVX2__)
template <typename T, arch_t ARCH>
static F_INLINE __m256 min(const __m256& x, const __m256& y)
{
    return _mm256_min_ps(x, y);
}

template <typename T, arch_t ARCH>
static F_INLINE __m256i min(const __m256i& x, const __m256i& y)
{
    return _mm256_min_epu16(x, y);
}
template <>
F_INLINE __m256i min<uint8_t, arch_t::USE_AVX2>(const __m256i& x, const __m256i& y)
{
    return _mm256_min_epu8(x, y);
}
#endif // __AVX2__
#endif // __SSE4_1__

/***************************** ABS_DIFF *************************************/
template <typename T, typename V>
static F_INLINE V abs_diff(const V& x, const V& y)
{
    return or_reg(sub<T>(x, y), sub<T>(y, x));
}

template <>
F_INLINE __m128 abs_diff<float>(const __m128& x, const __m128& y)
{
    return _mm_sub_ps(_mm_max_ps(x, y), _mm_min_ps(x, y));
}
#if defined(__AVX2__)
template <>
F_INLINE __m256 abs_diff<float>(const __m256& x, const __m256& y)
{
    return _mm256_sub_ps(_mm256_max_ps(x, y), _mm256_min_ps(x, y));
}
#endif // __AVX2__

/***************************** CLAMP **************************************/
template <typename T, typename V, arch_t ARCH>
static F_INLINE V clamp(const V& val, const V& minimum, const V& maximum)
{
    return min<T, ARCH>(max<T, ARCH>(val, minimum), maximum);
}

/******************************* AVERAGE4 **************************************/
template <typename T>
static F_INLINE __m128i average(const __m128i& x, const __m128i& y)
{
    return _mm_avg_epu16(x, y);
}
template <>
F_INLINE __m128i average<uint8_t>(const __m128i& x, const __m128i& y)
{
    return _mm_avg_epu8(x, y);
}

#if defined(__AVX2__)
template <typename T>
static F_INLINE __m256i average(const __m256i& x, const __m256i& y)
{
    return _mm256_avg_epu16(x, y);
}
template <>
F_INLINE __m256i average<uint8_t>(const __m256i& x, const __m256i& y)
{
    return _mm256_avg_epu8(x, y);
}
#endif

template <typename T, typename V>
static F_INLINE V get_avg(const V& a, const V& b, const V& x, const V& q)
{
    V t0 = sub<T>(average<T>(a, b), q);
    return average<T>(t0, x);
}

template <>
F_INLINE __m128
get_avg<float, __m128>(const __m128& a, const __m128& b, const __m128& x, const __m128& q)
{
    // __m128 q = _mm_set1_ps(0.25f);
    __m128 t = _mm_add_ps(_mm_add_ps(a, b), _mm_add_ps(x, x));
    return _mm_mul_ps(t, q);
}

#if defined(__AVX2__)
template <>
F_INLINE __m256
get_avg<float, __m256>(const __m256& a, const __m256& b, const __m256& x, const __m256& q)
{
    // __m128 q = _mm_set1_ps(0.25f);
    __m256 t = _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(x, x));
    return _mm256_mul_ps(t, q);
}
#endif

/****************************** BLENDV *************************/
template <arch_t ARCH>
static F_INLINE __m128
blendv(const __m128& x, const __m128& y, const __m128& mask)
{
    return _mm_or_ps(_mm_and_ps(mask, y), _mm_andnot_ps(mask, x));
}
template <arch_t ARCH>
static F_INLINE __m128i
blendv(const __m128i& x, const __m128i& y, const __m128i& mask)
{
    return _mm_or_si128(_mm_and_si128(mask, y), _mm_andnot_si128(mask, y));
}
#if defined(__SSE4_1__)
template <>
F_INLINE __m128
blendv<arch_t::USE_SSE41>(const __m128& x, const __m128& y, const __m128& mask)
{
    return _mm_blendv_ps(x, y, mask);
}
template <>
F_INLINE __m128i
blendv<arch_t::USE_SSE41>(const __m128i& x, const __m128i& y, const __m128i& mask)
{
    return _mm_blendv_epi8(x, y, mask);
}
#if defined(__AVX2__)
template <arch_t ARCH>
static F_INLINE __m256
blendv(const __m256& x, const __m256& y, const __m256& mask)
{
    return _mm256_blendv_ps(x, y, mask);
}
template <arch_t ARCH>
static F_INLINE __m256i
blendv(const __m256i& x, const __m256i& y, const __m256i& mask)
{
    return _mm256_blendv_epi8(x, y, mask);
}
#endif // __AVX2__
#endif // __SSE4_1__

#endif // __SSE2__

#endif
