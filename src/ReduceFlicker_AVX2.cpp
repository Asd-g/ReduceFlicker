#include <immintrin.h>

#include "ReduceFlicker.h"

/********************* LOAD ****************************************/
template <typename V> static F_INLINE V load(const uint8_t* p);

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

/********************* STORE *************************************/
static F_INLINE void stream(uint8_t* p, const __m256i& x)
{
    _mm256_stream_si256(reinterpret_cast<__m256i*>(p), x);
}

static F_INLINE void stream(uint8_t* p, const __m256& x)
{
    _mm256_stream_ps(reinterpret_cast<float*>(p), x);
}

/************************ SETZERO *********************************/
template <typename V> static F_INLINE V setzero();

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

/*********************** SET1 *************************************/
template <typename T, typename V> static F_INLINE V set1();

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

/********************* BIT OR *************************************/
static F_INLINE __m256i or_reg(const __m256i& x, const __m256i& y)
{
    return _mm256_or_si256(x, y);
}

/********************* BIT AND *************************************/
static F_INLINE __m256 and_reg(const __m256& x, const __m256& y)
{
    return _mm256_and_ps(x, y);
}
static F_INLINE __m256i and_reg(const __m256i& x, const __m256i& y)
{
    return _mm256_and_si256(x, y);
}

/********************* BIT ANDNOT *********************************/
static F_INLINE __m256 andnot_reg(const __m256& x, const __m256& y)
{
    return _mm256_andnot_ps(x, y);
}
static F_INLINE __m256i andnot_reg(const __m256i& x, const __m256i& y)
{
    return _mm256_andnot_si256(x, y);
}

/************************ COMPEQ *********************************/
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

/********************** SUB **************************************/
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

/************************** ADD *************************************/
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

/************************ MAX ************************************/
template <typename T>
static F_INLINE __m256 max(const __m256& x, const __m256& y)
{
    return _mm256_max_ps(x, y);
}

template <typename T>
static F_INLINE __m256i max(const __m256i& x, const __m256i& y)
{
    return _mm256_max_epu16(x, y);
}
template <>
F_INLINE __m256i max<uint8_t>(const __m256i& x, const __m256i& y)
{
    return _mm256_max_epu8(x, y);
}

/************************ MIN ************************************/
template <typename T>
static F_INLINE __m256 min(const __m256& x, const __m256& y)
{
    return _mm256_min_ps(x, y);
}

template <typename T>
static F_INLINE __m256i min(const __m256i& x, const __m256i& y)
{
    return _mm256_min_epu16(x, y);
}
template <>
F_INLINE __m256i min<uint8_t>(const __m256i& x, const __m256i& y)
{
    return _mm256_min_epu8(x, y);
}

/***************************** ABS_DIFF *************************************/
template <typename T, typename V>
static F_INLINE V abs_diff(const V& x, const V& y)
{
    return or_reg(sub<T>(x, y), sub<T>(y, x));
}

template <>
F_INLINE __m256 abs_diff<float>(const __m256& x, const __m256& y)
{
    return _mm256_sub_ps(_mm256_max_ps(x, y), _mm256_min_ps(x, y));
}

/***************************** CLAMP **************************************/
template <typename T, typename V>
static F_INLINE V clamp(const V& val, const V& minimum, const V& maximum)
{
    return min<T>(max<T>(val, minimum), maximum);
}

/******************************* AVERAGE4 **************************************/
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

template <typename T, typename V>
static F_INLINE V get_avg(const V& a, const V& b, const V& x, const V& q)
{
    V t0 = sub<T>(average<T>(a, b), q);
    return average<T>(t0, x);
}

template <>
F_INLINE __m256
get_avg<float, __m256>(const __m256& a, const __m256& b, const __m256& x, const __m256& q)
{
    __m256 t = _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(x, x));
    return _mm256_mul_ps(t, q);
}

/****************************** BLENDV *************************/
static F_INLINE __m256
blendv(const __m256& x, const __m256& y, const __m256& mask)
{
    return _mm256_blendv_ps(x, y, mask);
}

static F_INLINE __m256i
blendv(const __m256i& x, const __m256i& y, const __m256i& mask)
{
    return _mm256_blendv_epi8(x, y, mask);
}

template <typename T, int STRENGTH>
void proc_avx2(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept
{
    using V = std::conditional_t<std::is_integral_v<T>, __m256i, __m256>;

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

template void proc_avx2<uint8_t, 1>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_avx2<uint16_t, 1>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_avx2<float, 1>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;

template void proc_avx2<uint8_t, 2>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_avx2<uint16_t, 2>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_avx2<float, 2>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;

template void proc_avx2<uint8_t, 3>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_avx2<uint16_t, 3>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_avx2<float, 3>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;

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
void proc_a_avx2(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept
{
    using V = std::conditional_t<std::is_integral_v<T>, __m256i, __m256>;

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

template void proc_a_avx2<uint8_t, 1>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_a_avx2<uint16_t, 1>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_a_avx2<float, 1>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;

template void proc_a_avx2<uint8_t, 2>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_a_avx2<uint16_t, 2>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_a_avx2<float, 2>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;

template void proc_a_avx2<uint8_t, 3>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_a_avx2<uint16_t, 3>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
template void proc_a_avx2<float, 3>(uint8_t* dstp, const uint8_t* currp, const uint8_t** prevp, const uint8_t** nextp, int dstride, int cstride, int* pstride, int* nstride, int width, int height) noexcept;
