#include <iostream>
#include <mpi.h>
#include "JobTracker.hpp"
#include "TaskTracker.hpp"

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
    // Arguments check
    assert(argc == 8);

    // MPI init
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // CPU num
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int num_cpus = CPU_COUNT(&cpu_set);

    // Log arguments
    std::ofstream log(argv[7] + std::string(argv[1]) + "-log.out");
    assert(log.is_open());
    log << time(nullptr) << ",";
    log << "Start_Job,";
    log << argv[1] << ",";
    log << size << ",";
    log << num_cpus << ",";
    log << argv[2] << ",";
    log << argv[3] << ",";
    log << argv[4] << ",";
    log << argv[5] << ",";
    log << argv[6] << ",";
    log << argv[7] << std::endl;
    log.close();

    if(rank == 0) {
        JobTracker job_tracker(size, atoi(argv[2]), argv[1], argv[6], argv[7]);
        job_tracker.dispatch_map_tasks();
        // job_tracker.verify_ir();
        job_tracker.shuffle();
        // job_tracker.verify_shuffle();
        job_tracker.dispatch_reduce_tasks();
        // job_tracker.verify_reduce();
        job_tracker.finish();
    } else {
        TaskTracker task_tracker(rank, atoi(argv[5]), atoi(argv[3]), atoi(argv[2]), argv[1], argv[4], argv[7]);
        task_tracker.req_map_tasks();
        task_tracker.req_reduce_tasks();
    }

    return 0;
}