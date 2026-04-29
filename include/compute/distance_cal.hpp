#pragma once
#include "root.hpp"

namespace minihypervec
{
  namespace compute
  {

    static inline bool cpu_has_avx512bw()
    {
      return __builtin_cpu_supports("avx512f") &&
             __builtin_cpu_supports("avx512bw");
    }
    static inline bool cpu_has_avx512f()
    {
      return __builtin_cpu_supports("avx512f");
    }
    static inline bool cpu_has_avx2() { return __builtin_cpu_supports("avx2"); }
    static inline bool cpu_has_fma() { return __builtin_cpu_supports("fma"); }

#ifndef HV_PREFETCH_HINT
#define HV_PREFETCH_HINT _MM_HINT_T0
#endif

    static inline void hv_prefetch(const void *p)
    {
      _mm_prefetch(reinterpret_cast<const char *>(p), HV_PREFETCH_HINT);
    }

#ifndef HV_PFD_I8_AVX2
#define HV_PFD_I8_AVX2 256u
#endif
#ifndef HV_PFD_I8_AVX512
#define HV_PFD_I8_AVX512 256u
#endif

    static inline int64_t hsum_epi32_avx2(__m256i v)
    {
      alignas(32) int32_t buf[8];
      _mm256_store_si256(reinterpret_cast<__m256i *>(buf), v);
      int64_t s = 0;
      for (int i = 0; i < 8; ++i)
        s += buf[i];
      return s;
    }

    __attribute__((target("avx512f"))) static inline int64_t hsum_epi32_avx512(
        __m512i v)
    {
      alignas(64) int32_t buf[16];
      _mm512_store_si512(reinterpret_cast<void *>(buf), v);
      int64_t s = 0;
      for (int i = 0; i < 16; ++i)
        s += buf[i];
      return s;
    }

    static inline float hsum_ps_avx2(__m256 v)
    {
      alignas(32) float buf[8];
      _mm256_store_ps(buf, v);
      return buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
    }

    __attribute__((target("avx512f"))) static inline float hsum_ps_avx512(
        __m512 v)
    {
      alignas(64) float buf[16];
      _mm512_store_ps(buf, v);
      float s = 0.0f;
      for (int i = 0; i < 16; ++i)
        s += buf[i];
      return s;
    }

    __attribute__((target("avx2"))) static inline void InnerProductInt8InBatch_AVX2(
        const int8_t *x, const int8_t *y, int32_t *out, uint32_t dim,
        uint64_t batch)
    {
      const uint64_t D = (uint64_t)dim;
      uint64_t j = 0;

      for (; j + 3 < batch; j += 4)
      {
        const int8_t *y0 = y + (j + 0) * D;
        const int8_t *y1 = y + (j + 1) * D;
        const int8_t *y2 = y + (j + 2) * D;
        const int8_t *y3 = y + (j + 3) * D;

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();

        uint32_t i = 0;
        for (; i + 32 <= dim; i += 32)
        {
          const uint32_t pf = i + (uint32_t)HV_PFD_I8_AVX2;
          if (pf < dim)
          {
            hv_prefetch(x + pf);
            hv_prefetch(y0 + pf);
            hv_prefetch(y1 + pf);
            hv_prefetch(y2 + pf);
            hv_prefetch(y3 + pf);
          }

          __m128i x_lo8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + i));
          __m128i x_hi8 =
              _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + i + 16));
          __m256i x_lo16 = _mm256_cvtepi8_epi16(x_lo8);
          __m256i x_hi16 = _mm256_cvtepi8_epi16(x_hi8);

          auto step_y = [&](const int8_t *yy, __m256i &acc)
          {
            __m128i y_lo8 =
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(yy + i));
            __m128i y_hi8 =
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(yy + i + 16));
            __m256i y_lo16 = _mm256_cvtepi8_epi16(y_lo8);
            __m256i y_hi16 = _mm256_cvtepi8_epi16(y_hi8);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x_lo16, y_lo16));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x_hi16, y_hi16));
          };

          step_y(y0, acc0);
          step_y(y1, acc1);
          step_y(y2, acc2);
          step_y(y3, acc3);
        }

        int64_t s0 = hsum_epi32_avx2(acc0);
        int64_t s1 = hsum_epi32_avx2(acc1);
        int64_t s2 = hsum_epi32_avx2(acc2);
        int64_t s3 = hsum_epi32_avx2(acc3);

        for (; i < dim; ++i)
        {
          int xi = (int)x[i];
          s0 += xi * (int)y0[i];
          s1 += xi * (int)y1[i];
          s2 += xi * (int)y2[i];
          s3 += xi * (int)y3[i];
        }

        out[j + 0] = (int32_t)s0;
        out[j + 1] = (int32_t)s1;
        out[j + 2] = (int32_t)s2;
        out[j + 3] = (int32_t)s3;
      }

      for (; j < batch; ++j)
      {
        const int8_t *yj = y + j * D;
        __m256i acc = _mm256_setzero_si256();
        uint32_t i = 0;
        for (; i + 32 <= dim; i += 32)
        {
          const uint32_t pf = i + (uint32_t)HV_PFD_I8_AVX2;
          if (pf < dim)
          {
            hv_prefetch(x + pf);
            hv_prefetch(yj + pf);
          }

          __m128i x_lo8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + i));
          __m128i x_hi8 =
              _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + i + 16));
          __m128i y_lo8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(yj + i));
          __m128i y_hi8 =
              _mm_loadu_si128(reinterpret_cast<const __m128i *>(yj + i + 16));

          __m256i x_lo16 = _mm256_cvtepi8_epi16(x_lo8);
          __m256i x_hi16 = _mm256_cvtepi8_epi16(x_hi8);
          __m256i y_lo16 = _mm256_cvtepi8_epi16(y_lo8);
          __m256i y_hi16 = _mm256_cvtepi8_epi16(y_hi8);

          acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x_lo16, y_lo16));
          acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x_hi16, y_hi16));
        }
        int64_t s = hsum_epi32_avx2(acc);
        for (; i < dim; ++i)
          s += (int)x[i] * (int)yj[i];
        out[j] = (int32_t)s;
      }
    }

    __attribute__((target("avx512f,avx512bw"))) static inline void
    InnerProductInt8InBatch_AVX512(const int8_t *x, const int8_t *y, int32_t *out,
                                   uint32_t dim, uint64_t batch)
    {
      const uint64_t D = (uint64_t)dim;
      uint64_t j = 0;

      for (; j + 7 < batch; j += 8)
      {
        const int8_t *yptr[8] = {y + (j + 0) * D, y + (j + 1) * D, y + (j + 2) * D,
                                 y + (j + 3) * D, y + (j + 4) * D, y + (j + 5) * D,
                                 y + (j + 6) * D, y + (j + 7) * D};

        __m512i acc[8];
        for (int t = 0; t < 8; ++t)
          acc[t] = _mm512_setzero_si512();

        uint32_t i = 0;
        for (; i + 64 <= dim; i += 64)
        {
          const uint32_t pf = i + (uint32_t)HV_PFD_I8_AVX512;
          if (pf < dim)
          {
            hv_prefetch(x + pf);
            for (int t = 0; t < 8; ++t)
              hv_prefetch(yptr[t] + pf);
          }

          __m256i x_lo8 =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x + i));
          __m256i x_hi8 =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x + i + 32));
          __m512i x_lo16 = _mm512_cvtepi8_epi16(x_lo8);
          __m512i x_hi16 = _mm512_cvtepi8_epi16(x_hi8);

          for (int t = 0; t < 8; ++t)
          {
            __m256i y_lo8 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(yptr[t] + i));
            __m256i y_hi8 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(yptr[t] + i + 32));
            __m512i y_lo16 = _mm512_cvtepi8_epi16(y_lo8);
            __m512i y_hi16 = _mm512_cvtepi8_epi16(y_hi8);

            acc[t] = _mm512_add_epi32(acc[t], _mm512_madd_epi16(x_lo16, y_lo16));
            acc[t] = _mm512_add_epi32(acc[t], _mm512_madd_epi16(x_hi16, y_hi16));
          }
        }

        for (int t = 0; t < 8; ++t)
        {
          int64_t s = hsum_epi32_avx512(acc[t]);
          for (uint32_t k = i; k < dim; ++k)
            s += (int)x[k] * (int)yptr[t][k];
          out[j + t] = (int32_t)s;
        }
      }

      for (; j < batch; ++j)
      {
        const int8_t *yj = y + j * D;
        __m512i acc = _mm512_setzero_si512();
        uint32_t i = 0;
        for (; i + 64 <= dim; i += 64)
        {
          const uint32_t pf = i + (uint32_t)HV_PFD_I8_AVX512;
          if (pf < dim)
          {
            hv_prefetch(x + pf);
            hv_prefetch(yj + pf);
          }

          __m256i x_lo8 =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x + i));
          __m256i x_hi8 =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x + i + 32));
          __m256i y_lo8 =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(yj + i));
          __m256i y_hi8 =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(yj + i + 32));
          __m512i x_lo16 = _mm512_cvtepi8_epi16(x_lo8);
          __m512i x_hi16 = _mm512_cvtepi8_epi16(x_hi8);
          __m512i y_lo16 = _mm512_cvtepi8_epi16(y_lo8);
          __m512i y_hi16 = _mm512_cvtepi8_epi16(y_hi8);
          acc = _mm512_add_epi32(acc, _mm512_madd_epi16(x_lo16, y_lo16));
          acc = _mm512_add_epi32(acc, _mm512_madd_epi16(x_hi16, y_hi16));
        }
        int64_t s = hsum_epi32_avx512(acc);
        for (; i < dim; ++i)
          s += (int)x[i] * (int)yj[i];
        out[j] = (int32_t)s;
      }
    }

    inline void InnerProductInt8InBatch(const int8_t *x, const int8_t *y,
                                        int32_t *out, uint32_t dim,
                                        uint64_t batch)
    {
      if (!x || !y || !out || dim == 0 || batch == 0)
        return;

      if (cpu_has_avx512bw())
      {
        InnerProductInt8InBatch_AVX512(x, y, out, dim, batch);
      }
      else
      {
        InnerProductInt8InBatch_AVX2(x, y, out, dim, batch);
      }
    }

  } // namespace compute
} // namespace minihypervec
