#include <iostream>
#include <mpi.h>
#include "JobTracker.hpp"

void print_args(char *argv[]) {
    std::cout << "JOB_NAME: " << argv[1] << std::endl;
    std::cout << "NUM_REDUCER: " << argv[2] << std::endl;
    std::cout << "DELAY: " << argv[3] << std::endl;
    std::cout << "INPUT_FILENAME: " << argv[4] << std::endl;
    std::cout << "CHUNK_SIZE: " << argv[5] << std::endl;
    std::cout << "LOCALITY_CONFIG_FILENAME: " << argv[6] << std::endl;
    std::cout << "OUTPUT_DIR: " << argv[7] << std::endl;
}

int main(int argc, char *argv[]) {
    // Arguments count
    assert(argc == 8);    

    // MPI init
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0) {
        std::cout << "Job tracker: " << rank << std::endl;
        print_args(argv);
        JobTracker job_tracker(size - 1, argv[6]);
    } else {
        std::cout << "Task tracker: " << rank << std::endl;
    }

    return 0;
}