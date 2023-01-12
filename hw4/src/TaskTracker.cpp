#include "TaskTracker.hpp"

// Constructor
TaskTracker::TaskTracker(int node_id, int chunk_size, std::string word_file) {
    this->node_id = node_id;
    this->chunk_size = chunk_size;
    this->word_file = word_file;
    set_num_cpus();
    print();
}

// Methods
void TaskTracker::set_num_cpus() {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_cpus = CPU_COUNT(&cpu_set);
}

// Utils
void TaskTracker::print() {
    std::cout << "[TaskTracker]" << std::endl;
    std::cout << "num_cpus: " << num_cpus << std::endl;
    std::cout << "node_id: " << node_id << std::endl;
    std::cout << "chunk_size: " << chunk_size << std::endl;
    std::cout << "word_file: " << word_file << std::endl;
}