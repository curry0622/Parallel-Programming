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

void TaskTracker::input_split(int chunk_id) {
    std::ifstream fin(word_file);
    std::string line;
    int line_id = 0;

    // Ignore lines
    for(int i = 1; i < chunk_id; i++) {
        for(int j = 0; j < chunk_size; j++) {
            std::getline(fin, line);
            line_id++;
        }
    }

    // Read lines in chunk[chunk_id]
    for(int i = 0; i < chunk_size; i++) {
        std::getline(fin, line);
        line_id++;
        records[line_id] = line;
        std::cout << line_id << ": " << std::endl;
    }

    fin.close();
}

void TaskTracker::map(int line_id) {
    std::map<std::string, int> word_count;
    std::stringstream ss(records[line_id]);
    std::string word;
    while (ss >> word) {
        word_count[word]++;
    }
}

// Utils
void TaskTracker::print() {
    std::cout << "[TaskTracker]" << std::endl;
    std::cout << "num_cpus: " << num_cpus << std::endl;
    std::cout << "node_id: " << node_id << std::endl;
    std::cout << "chunk_size: " << chunk_size << std::endl;
    std::cout << "word_file: " << word_file << std::endl;
}