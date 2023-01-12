#include "TaskTracker.hpp"

// Constructor
TaskTracker::TaskTracker(int node_id, int chunk_size, std::string word_file) {
    this->node_id = node_id;
    this->chunk_size = chunk_size;
    this->word_file = word_file;
    set_num_cpus();
    while(true) {
        if(request_map_task() == -1)
            break;
    }
}

// Methods
int TaskTracker::request_map_task() {
    // Send node_id to job tracker to request a map task
    // tag 0: request map task
    MPI_Send(&node_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    std::cout << "TaskTracker[" << node_id << "] MPI_Send to JobTracker requests map task" << std::endl;

    // Receive chunk_id from job tracker
    // tag 1: receive chunk_id & remote or not
    int buffer[2] = {0, 0};
    MPI_Recv(buffer, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::cout << "TaskTracker[" << node_id << "] MPI_Recv from JobTracker chunk_id = " << buffer[0] << ", remote = " << buffer[1] << std::endl;

    if(buffer[0] == -1) {
        std::cout << "TaskTracker[" << node_id << "] receives chunk_id = -1, exit" << std::endl;
        return -1;
    }

    return buffer[0];
}

void TaskTracker::set_num_cpus() {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_cpus = CPU_COUNT(&cpu_set);
}

void TaskTracker::input_split(int chunk_id) {
    // This function reads a data chunk from the input file,
    // and splits it into a set of records.
    // Each record is a line in the input file.
    // The key of the record is the line number.
    // The value of the record is the line content.

    // Variables
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
    }

    fin.close();
}

void TaskTracker::map(int line_id) {
    // It reads an input key-value pair record
    // and output to a set of intermediate key-value pairs.
    // The key of the intermediate key-value pair is a word.
    // The value of the intermediate key-value pair is the count of the word.

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