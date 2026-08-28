#ifndef DISTANCE_H
#define DISTANCE_H

#include <cstdint>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>

#define square(x) ((x)*(x))


// Base L2 vector distance with scalar serial execution
static float l2sq_scalar(const float* a, const float* b, uint32_t dim)
{
    float result = 0.0f;
    for (uint32_t k = 0; k < dim; k++) {result += square(a[k] - b[k]);}
    return result;
}

// Calculating squared L2 vector distance with AVX-512
static float l2sq_avx512()
{

}

#endif /* DISTANCE_H */