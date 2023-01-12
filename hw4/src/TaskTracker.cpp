#include "TaskTracker.hpp"

// Constructor
TaskTracker::TaskTracker(int node_id, int chunk_size, int delay, int num_reducers, std::string word_file) {
    this->node_id = node_id;
    this->chunk_size = chunk_size;
    this->delay = delay;
    this->num_reducers = num_reducers;
    this->word_file = word_file;
    set_num_cpus();
}

// Methods
void TaskTracker::req_map_tasks() {
    // Create map threads
    int num_map_threads = num_cpus - 1;
    int thread_id[num_map_threads];
    pthread_t map_threads[num_map_threads];

    for(int i = 0; i < num_map_threads; i++) {
        thread_id[i] = i;
        pthread_create(&map_threads[i], NULL, map_thread_func, (void*)&thread_id[i]);
    }
}

void* TaskTracker::map_thread_func(void* thread_id) {
    return NULL;
}

int TaskTracker::get_chunk_id() {
    // Send node_id to job tracker to request a map task using tag[0]
    MPI_Send(&node_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    std::cout << "TaskTracker[" << node_id << "] MPI_Send to JobTracker requests map task" << std::endl;

    // Receive chunk_id & remote from job tracker using tag[1]
    int buffer[2] = {0, 0};
    MPI_Recv(buffer, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::cout << "TaskTracker[" << node_id << "] MPI_Recv from JobTracker chunk_id = " << buffer[0] << ", remote = " << buffer[1] << std::endl;

    // Return chunk_id
    return buffer[0];
}

void TaskTracker::set_num_cpus() {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_cpus = CPU_COUNT(&cpu_set);
}

std::map<int, std::string> TaskTracker::input_split(int chunk_id) {
    // This function reads a data chunk from the input file,
    // and splits it into a set of records.
    // Each record is a line in the input file.
    // The key of the record is the line number.
    // The value of the record is the line content.

    // Variables
    std::map<int, std::string> records;
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
    return records;
}

std::map<std::string, int> TaskTracker::map(std::pair<int, std::string> record) {
    // It reads an input key-value pair record
    // and output to a set of intermediate key-value pairs.
    // The key of the intermediate key-value pair is a word.
    // The value of the intermediate key-value pair is the count of the word.

    std::map<std::string, int> word_count;
    std::stringstream ss(record.second);
    std::string word;
    while (ss >> word) {
        word_count[word]++;
    }
    return word_count;
}

int TaskTracker::partition(std::string key) {
    int offset = key[0] - 'A';
    return offset % num_reducers;
}

// Utils
void TaskTracker::print() {
    std::cout << "[TaskTracker]" << std::endl;
    std::cout << "num_cpus: " << num_cpus << std::endl;
    std::cout << "node_id: " << node_id << std::endl;
    std::cout << "chunk_size: " << chunk_size << std::endl;
    std::cout << "word_file: " << word_file << std::endl;
}