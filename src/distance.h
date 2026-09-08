#ifndef DISTANCE_H
#define DISTANCE_H

#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#define square(x) ((x)*(x))

// -------------------------------------------------------------------------------- //
// "Templating" the distance function type
// -------------------------------------------------------------------------------- //
typedef float (*dist_fn_t)(const float *a, const float *b, uint32_t dim);

// -------------------------------------------------------------------------------- //
// Metric declerations
// -------------------------------------------------------------------------------- //
static inline float l2_dist(const float* a, const float* b, uint32_t dim);
static inline float cosine_dist(const float* a, const float* b, uint32_t dim);


// -------------------------------------------------------------------------------- //
// Compile-time dispatch the choice of distance
// -------------------------------------------------------------------------------- //
static inline dist_fn_t metric()
{
    #if defined (L2_IMPLEMENTATION)
        return l2_dist;
    #elif defined (COSINE_IMPLEMENTATION)
        return cosine_dist;
    #else
        printf("No metric implementation specified! Default to L2 distance.\n");
        return  l2_dist;
    #endif
}

// -------------------------------------------------------------------------------- //
// Helpers
// -------------------------------------------------------------------------------- //
static inline float horizontal_sum_reduce_avx256(__m256 v) 
{
    // AVX-256 does not have a sane single intrinsics to do this so hand-writing a horizontal-recudtion scheme
    __m128 lo = _mm256_extractf128_ps(v, 0);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

// -------------------------------------------------------------------------------- //
// L2 metric implementations
// -------------------------------------------------------------------------------- //
static inline float l2sq_avx256(const float* a, const float* b, uint32_t dim);
static inline float l2sq_avx512(const float* a, const float* b, uint32_t dim);
static inline float l2sq_scalar(const float* a, const float* b, uint32_t dim);

static inline float l2sq_dist(const float* a, const float* b, uint32_t dim)
{
    #if defined (__AVX512F__)
        return l2sq_avx512(a, b, dim);
    #elif defined (__AVX2__) && defined (__FMA__)
        return l2sq_avx256(a, b, dim);
    #else
        return l2sq_scalar(a, b, dim);
    #endif
}

static inline float l2_dist(const float* a, const float* b, uint32_t dim) {return sqrtf(l2sq_dist(a, b, dim));}


// Base L2 vector distance with scalar serial execution
static inline float l2sq_scalar(const float* a, const float* b, uint32_t dim)
{
    #if defined(DEBUG)
        printf("Calculating L2 serially...\n");
    #endif

    float result = 0.0f;
    for (uint32_t k = 0; k < dim; k++) {result += square(a[k] - b[k]);}
    return result;
}

// Calculating squared L2 vector distance with AVX-512
static inline float l2sq_avx512(const float* a, const float* b, uint32_t dim)
{
    #if defined(DEBUG)
        printf("Calculating L2 with AVX-512...\n");
    #endif

    __m512 accumulator = _mm512_setzero_ps();
    
    // Note 512-bits is 16 floats so jump by that amount
    uint32_t k = 0;
    for (; k + 16 <= dim; k += 16) {
        // Load data into 512-bit wide registers
        __m512 va = _mm512_loadu_ps(a + k);
        __m512 vb = _mm512_loadu_ps(b + k);

        // Accumulate the (vectorized) difference
        __m512 diff = _mm512_sub_ps(va, vb);
        accumulator = _mm512_fmadd_ps(diff, diff, accumulator); /* acc += diff^2 */
    }

    // Reduce the result
    float result = _mm512_reduce_add_ps(accumulator);

    // Accumulate tail-elements contribution
    for (; k < dim; k++) {result += square(a[k] - b[k]);}

    return result;
}

// Calculating squared L2 vector distance with AVX-256
static inline float l2sq_avx256(const float* a, const float* b, uint32_t dim)
{
    #if defined(DEBUG)
        printf("Calculating L2 with AVX-256...\n");
    #endif

    __m256 accumulator = _mm256_setzero_ps();

    // Accumulate results via 8 floats (256-bits) chunks
    uint32_t k = 0;
    for (; k + 8 <= dim; k+= 8) {
        // Load data into 256-bit wide registers
        __m256 va = _mm256_loadu_ps(a + k);
        __m256 vb = _mm256_loadu_ps(b + k);

        // Accumulate the (vectorized) difference
        __m256 diff = _mm256_sub_ps(va, vb);
        accumulator = _mm256_fmadd_ps(diff, diff, accumulator); /* acc += diff^2 */
    
    }

    // Reduction 
    float result = horizontal_sum_reduce_avx256(accumulator);

    // Accumulate tail-elements contribution
    for (; k < dim; k++) {result += square(a[k] - b[k]);}

    return result;
}

// -------------------------------------------------------------------------------- //
// Cosine metric implementations
// -------------------------------------------------------------------------------- //
static inline float cosine_avx256(const float* a, const float* b, uint32_t dim);
static inline float cosine_avx512(const float* a, const float* b, uint32_t dim);
static inline float cosine_scalar(const float* a, const float* b, uint32_t dim);

static inline float cosine_dist(const float* a, const float* b, uint32_t dim)
{
    #if defined (__AVX512F__)
        return cosine_avx512(a, b, dim);
    #elif defined (__AVX2__) && defined (__FMA__)
        return cosine_avx256(a, b, dim);
    #else
        return cosine_scalar(a, b, dim);
    #endif
}

static inline float cosine_scalar(const float* a, const float* b, uint32_t dim)
{
    #if defined(DEBUG)
        printf("Calculating Cosine serially...\n");
    #endif

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    // Calculating cosine similarity
    for (uint32_t k = 0; k < dim; k++) {
        dot += a[k] * b[k];
        norm_a += square(a[k]);
        norm_b += square(b[k]);
    }

    float norms = norm_a * norm_b;
    if (norms == 0.0f) {
        return 1.0f; // Short-circuit to prevent zero-division
    }

    // Cosine distance := 1 - Cosine similarity
    return 1.0f - (dot / sqrtf(norms));
}

static inline float cosine_avx256(const float* a, const float* b, uint32_t dim)
{
    #if defined(DEBUG)
        printf("Calculating Cosine with AVX-256...\n");
    #endif

    __m256 acc_dot = _mm256_setzero_ps();
    __m256 acc_na = _mm256_setzero_ps();
    __m256 acc_nb = _mm256_setzero_ps();

    uint32_t k = 0;
    for (; k + 8 <= dim; k += 8) {
        __m256 va = _mm256_loadu_ps(a + k);
        __m256 vb = _mm256_loadu_ps(b + k);

        acc_dot = _mm256_fmadd_ps(va, vb, acc_dot);
        acc_na = _mm256_fmadd_ps(va, va, acc_na);
        acc_nb = _mm256_fmadd_ps(vb, vb, acc_nb);
    }

    // Reduction
    float dot = horizontal_sum_reduce_avx256(acc_dot);
    float norm_a = horizontal_sum_reduce_avx256(acc_na);
    float norm_b = horizontal_sum_reduce_avx256(acc_nb);

    // Accumulate tail-elements contribution
    for (; k < dim; k++) {
        dot += a[k] * b[k];
        norm_a += square(a[k]);
        norm_b += square(b[k]);
    }

    float norms = norm_a * norm_b;
    if (norms == 0.0f) return 1.0f;
    
    return 1.0f - (dot / sqrtf(norms));
}

static inline float cosine_avx512(const float* a, const float* b, uint32_t dim)
{
    #if defined(DEBUG)
        printf("Calculating Cosine with AVX-512...\n");
    #endif

    __m512 acc_dot = _mm512_setzero_ps();
    __m512 acc_na = _mm512_setzero_ps();
    __m512 acc_nb = _mm512_setzero_ps();
    
    uint32_t k = 0;
    for (; k + 16 <= dim; k += 16) {
        __m512 va = _mm512_loadu_ps(a + k);
        __m512 vb = _mm512_loadu_ps(b + k);

        acc_dot = _mm512_fmadd_ps(va, vb, acc_dot);
        acc_na = _mm512_fmadd_ps(va, va, acc_na);
        acc_nb = _mm512_fmadd_ps(vb, vb, acc_nb);
    }

    // Reduce the result
    float dot = _mm512_reduce_add_ps(acc_dot);
    float norm_a = _mm512_reduce_add_ps(acc_na);
    float norm_b = _mm512_reduce_add_ps(acc_nb);

    // Accumulate tail-elements contribution
    for (; k < dim; k++) {
        dot += a[k] * b[k];
        norm_a += square(a[k]);
        norm_b += square(b[k]);
    }

    float norms = norm_a * norm_b;
    if (norms == 0.0f) {
        return 1.0f;
    }
    
    return 1.0f - (dot / sqrtf(norms));
}


#endif /* DISTANCE_H */