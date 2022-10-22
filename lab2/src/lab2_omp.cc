#include <assert.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char** argv) {
    // Dummy check
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }

    // Get args
    unsigned long long r = atoll(argv[1]);
    unsigned long long rSqr = r * r;
    unsigned long long k = atoll(argv[2]);
    unsigned long long pixels = 0;

#pragma omp parallel for reduction(+:pixels)
    // Calculate pixels
    for (unsigned long long x = 0; x < r; x++) {
        unsigned long long y = ceil(sqrtl(rSqr - x*x));
        pixels += y;
    }

    // Print result
    printf("%llu\n", (4 * (pixels % k)) % k);
}
