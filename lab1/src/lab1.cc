#include <mpi.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	int rank, n;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n);

	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long num = 0, pixels = 0, sum = 0;

	if(r % n != 0) {
		if(rank != n - 1)
			num = r / n + 1;
		else
			num = r - (n - 1) * (r / n + 1);
	} else
		num = r / n;

	if(rank != n - 1) {
		for(unsigned long long x = rank * num; x < num * (rank + 1); x++) {
			pixels += ceil(sqrtl(r*r - x*x));
			pixels %= k;
		}
	} else {
		for(unsigned long long x = r - num; x < r; x++) {
			pixels += ceil(sqrtl(r*r - x*x));
			pixels %= k;
		}
	}

	// printf("rank: %d, num: %llu, pixels: %llu, from: %llu, to: %llu\n", rank, num, pixels, rank * num, num * (rank + 1));
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&pixels, &sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	if(rank == 0)
		printf("%llu\n", (4 * sum) % k);

	MPI_Finalize();
}
