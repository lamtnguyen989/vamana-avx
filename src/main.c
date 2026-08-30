#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DEBUG
#include "distance.h"
#include "pq.h"
#include "vecfile.h"

int main(int argc, char** argv)
{
    uint32_t dim = 132;
    float* a = malloc(dim*sizeof(float));
    float* b = malloc(dim*sizeof(float));

    for (uint32_t k =0; k < dim; k++) {
        a[k] = k;
        b[k] = k+1;
    }

    dist_fn_t dist = metric();
    printf("%f\n", dist(a, b, dim));

    free(a);
    free(b);

    return 0;
}