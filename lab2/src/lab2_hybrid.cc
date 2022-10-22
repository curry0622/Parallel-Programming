#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
    // MPI init
    int mpiRank, mpiSize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // Dummy check
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }

    // Get args
    unsigned long long r = atoll(argv[1]);
    unsigned long long k = atoll(argv[2]);
    unsigned long long rSqr = r * r;
    unsigned long long pixels = 0;
    unsigned long long totalPixels = 0;

    // MPI range
    unsigned long long start = mpiRank * r / mpiSize;
    unsigned long long end = (mpiRank + 1) * r / mpiSize;

#pragma omp parallel for reduction(+:pixels)
    // Calculate pixels
    for (unsigned long long x = start; x < end; x++) {
        unsigned long long y = ceil(sqrtl(rSqr - x*x));
        pixels += y;
    }

    // MPI reduce
    MPI_Reduce(&pixels, &totalPixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Print result
    if(mpiRank == 0) {
        printf("%llu\n", (4 * (totalPixels % k)) % k);
    }
}
