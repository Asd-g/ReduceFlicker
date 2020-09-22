#include <emmintrin.h>

#include "ReduceFlicker.h"

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

/********************* STORE *************************************/
static F_INLINE void stream(uint8_t* p, const __m128i& x)
{
    _mm_stream_si128(reinterpret_cast<__m128i*>(p), x);
}

static F_INLINE void stream(uint8_t* p, const __m128& x)
{
    _mm_stream_ps(reinterpret_cast<float*>(p), x);
}

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

/********************* BIT OR *************************************/
static F_INLINE __m128i or_reg(const __m128i& x, const __m128i& y)
{
    return _mm_or_si128(x, y);
}

/********************* BIT AND *************************************/
static F_INLINE __m128 and_reg(const __m128& x, const __m128& y)
{
    return _mm_and_ps(x, y);
}
static F_INLINE __m128i and_reg(const __m128i& x, const __m128i& y)
{
    return _mm_and_si128(x, y);
}

/********************* BIT ANDNOT *********************************/
static F_INLINE __m128 andnot_reg(const __m128& x, const __m128& y)
{
    return _mm_andnot_ps(x, y);
}
static F_INLINE __m128i andnot_reg(const __m128i& x, const __m128i& y)
{
    return _mm_andnot_si128(x, y);
}

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

/************************ MAX ************************************/
template <typename T>
static F_INLINE __m128 max(const __m128& x, const __m128& y)
{
    return _mm_max_ps(x, y);
}

template <typename T>
static F_INLINE __m128i max(const __m128i& x, const __m128i& y)
{
    return _mm_adds_epu16(y, _mm_subs_epu16(x, y));
}

template <>
F_INLINE __m128i max<uint8_t>(const __m128i& x, const __m128i& y)
{
    return _mm_max_epu8(x, y);
}

/************************ MIN ************************************/
template <typename T>
static F_INLINE __m128 min(const __m128& x, const __m128& y)
{
    return _mm_min_ps(x, y);
}

template <typename T>
static F_INLINE __m128i min(const __m128i& x, const __m128i& y)
{
    return _mm_subs_epu16(x, _mm_subs_epu16(x, y));
}

template <>
F_INLINE __m128i min<uint8_t>(const __m128i& x, const __m128i& y)
{
    return _mm_min_epu8(x, y);
}

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

/***************************** CLAMP **************************************/
template <typename T, typename V>
static F_INLINE V clamp(const V& val, const V& minimum, const V& maximum)
{
    return min<T>(max<T>(val, minimum), maximum);
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

/****************************** BLENDV *************************/
static F_INLINE __m128
blendv(const __m128& x, const __m128& y, const __m128& mask)
{
    return _mm_or_ps(_mm_and_ps(mask, y), _mm_andnot_ps(mask, x));
}

static F_INLINE __m128i
blendv(const __m128i& x, const __m128i& y, const __m128i& mask)
{
    return _mm_or_si128(_mm_and_si128(mask, y), _mm_andnot_si128(mask, y));
}

template <typename T, int STRENGTH>
void proc_sse2(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept
{
    using V = std::conditional_t<std::is_integral_v<T>, __m128i, __m128>;

    const uint8_t* prv0, * prv1, * prv2, * nxt0, * nxt1, * nxt2;
    prv0 = prevp[0];
    prv1 = prevp[1];
    nxt0 = nextp[0];
    if (STRENGTH > 1)
    {
        nxt1 = nextp[1];
    }
    if (STRENGTH > 2)
    {
        prv2 = prevp[2];
        nxt2 = nextp[2];
    }

    width *= sizeof(T);

    V q = set1<T, V>();

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; x += sizeof(V))
        {
            const V curx = load<V>(currp + x);
            V d = abs_diff<T, V>(curx, load<V>(prv1 + x));
            if (STRENGTH > 1)
            {
                d = min<T>(d, abs_diff<T, V>(curx, load<V>(nxt1 + x)));
            }
            if (STRENGTH > 2)
            {
                d = min<T>(d, abs_diff<T, V>(curx, load<V>(prv2 + x)));
                d = min<T>(d, abs_diff<T, V>(curx, load<V>(nxt2 + x)));
            }
            const V pr0 = load<V>(prv0 + x);
            const V nx0 = load<V>(nxt0 + x);
            const V ul = max<T>(sub<T>(min<T>(pr0, nx0), d), curx);
            const V ll = min<T>(add<T>(max<T>(pr0, nx0), d), curx);
            const V avg = get_avg<T, V>(pr0, nx0, curx, q);
            stream(dstp + x, clamp<T, V>(avg, ll, ul));
        }
        prv0 += pstride[0];
        prv1 += pstride[1];
        nxt0 += nstride[0];
        currp += cstride;
        dstp += dstride;
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

template void proc_sse2<uint8_t, 1>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_sse2<uint16_t, 1>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_sse2<float, 1>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;

template void proc_sse2<uint8_t, 2>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_sse2<uint16_t,  2>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_sse2<float, 2>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;

template void proc_sse2<uint8_t, 3>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_sse2<uint16_t, 3>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_sse2<float, 3>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;

template <typename T, typename V>
static F_INLINE void
update_diff(const V& x, const V& y, V& d1, V& d2, const V& zero)
{
    const V maxxy = max<T>(x, y);
    const V mask = cmpeq<T>(x, maxxy);
    const V d = sub<T>(maxxy, min<T>(x, y));
    d1 = blendv(zero, min<T>(d, d1), mask);
    d2 = blendv(min<T>(d, d2), zero, mask);
}

template <typename T, int STRENGTH>
void proc_a_sse2(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept
{
    using V = std::conditional_t<std::is_integral_v<T>, __m128i, __m128>;

    const uint8_t* prv0, * prv1, * prv2, * nxt0, * nxt1, * nxt2;
    prv0 = prevp[0];
    prv1 = prevp[1];
    nxt0 = nextp[0];
    if (STRENGTH > 1)
    {
        nxt1 = nextp[1];
    }
    if (STRENGTH > 2)
    {
        prv2 = prevp[2];
        nxt2 = nextp[2];
    }

    width *= sizeof(T);

    V q = set1<T, V>();
    V zero = setzero<V>();

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; x += sizeof(V))
        {
            const V curx = load<V>(currp + x);
            V t0 = load<V>(prv1 + x);
            V t1 = max<T>(t0, curx);
            V t2 = cmpeq<T>(t0, t1);
            t0 = sub<T>(t1, min<T>(t0, curx));
            V d1 = and_reg(t2, t0);
            V d2 = andnot_reg(t2, t0);
            if (STRENGTH > 1)
            {
                update_diff<T, V>(load<V>(nxt1 + x), curx, d1, d2, zero);
            }
            if (STRENGTH > 2)
            {
                update_diff<T, V>(load<V>(prv2 + x), curx, d1, d2, zero);
                update_diff<T, V>(load<V>(nxt2 + x), curx, d1, d2, zero);
            }
            const V pr0 = load<V>(prv0 + x);
            const V nx0 = load<V>(nxt0 + x);
            const V ul = max<T>(sub<T>(min<T>(pr0, nx0), d1), curx);
            const V ll = min<T>(add<T>(max<T>(pr0, nx0), d2), curx);
            const V avg = get_avg<T, V>(pr0, nx0, curx, q);
            stream(dstp + x, clamp<T, V>(avg, ll, ul));
        }
        prv0 += pstride[0];
        prv1 += pstride[1];
        nxt0 += nstride[0];
        currp += cstride;
        dstp += dstride;
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

template void proc_a_sse2<uint8_t, 1>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_a_sse2<uint16_t, 1>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_a_sse2<float, 1>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;

template void proc_a_sse2<uint8_t, 2>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_a_sse2<uint16_t, 2>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_a_sse2<float, 2>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;

template void proc_a_sse2<uint8_t, 3>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_a_sse2<uint16_t, 3>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_a_sse2<float, 3>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
