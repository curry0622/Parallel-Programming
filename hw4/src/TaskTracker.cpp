#include "TaskTracker.hpp"

// Declare static variables
pthread_mutex_t TaskTracker::mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t TaskTracker::cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t TaskTracker::mutex2 = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t TaskTracker::cond2 = PTHREAD_COND_INITIALIZER;
std::queue<std::pair<int, int>> TaskTracker::tasks;
std::string TaskTracker::job_name = "";
std::string TaskTracker::word_file = "";
std::string TaskTracker::output_dir = "";
int TaskTracker::num_working = 0;
int TaskTracker::delay = 0;
int TaskTracker::chunk_size = 0;
int TaskTracker::node_id = 0;

// Constructor
TaskTracker::TaskTracker(int node_id, int chunk_size, int delay, int num_reducers, std::string job_name, std::string word_file, std::string output_dir) {
    this->node_id = node_id;
    this->chunk_size = chunk_size;
    this->delay = delay;
    this->num_reducers = num_reducers;
    this->job_name = job_name;
    this->word_file = word_file;
    this->output_dir = output_dir;
    set_num_cpus();
}

// Methods
void TaskTracker::req_map_tasks() {
    // Create map threads
    int num_map_threads = num_cpus - 1;
    pthread_t map_threads[num_map_threads];
    int map_thread_ids[num_map_threads];
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond, NULL);
    for(int i = 0; i < num_map_threads; i++) {
        map_thread_ids[i] = i;
        pthread_create(&map_threads[i], NULL, map_thread_func, (void*)&map_thread_ids[i]);
    }

    // While there are idle threads, get chunk_id from job tracker
    while(true) {
        // Wait until there is an idle thread
        pthread_mutex_lock(&mutex2);
        while(num_working == num_map_threads) {
            pthread_cond_wait(&cond2, &mutex2);
        }
        pthread_mutex_unlock(&mutex2);

        // Get task
        std::pair<int, int> task = get_task();

        // // Add task to queue
        // pthread_mutex_lock(&mutex);
        // tasks.push(task);
        // pthread_cond_signal(&cond); // Wake up a thread
        // pthread_mutex_unlock(&mutex);

        if(task.first == -1) {
            // join map threads
            for(int i = 0; i < num_map_threads; i++) {
                // Add task to queue
                pthread_mutex_lock(&mutex);
                tasks.push(task);
                pthread_cond_signal(&cond); // Wake up a thread
                pthread_mutex_unlock(&mutex);
            }
            for(int i = 0; i < num_map_threads; i++) {
                pthread_join(map_threads[i], NULL);
            }
            break;
        } else {
            // Add task to queue
            pthread_mutex_lock(&mutex);
            tasks.push(task);
            pthread_cond_signal(&cond); // Wake up a thread
            pthread_mutex_unlock(&mutex);
        }

        // Increment num_working
        pthread_mutex_lock(&mutex2);
        num_working++;
        pthread_mutex_unlock(&mutex2);
    }
}

void* TaskTracker::map_thread_func(void* id) {
    while(true) {
        // Get task
        pthread_mutex_lock(&mutex);
        while(tasks.empty()) {
            pthread_cond_wait(&cond, &mutex);
        }
        std::pair<int, int> task = tasks.front();
        if(task.first == -1) {
            tasks.pop();
            pthread_mutex_unlock(&mutex);
            pthread_exit(NULL);
        } else {
            tasks.pop();
        }
        pthread_mutex_unlock(&mutex);

        // If task is remote, sleep
        if(task.second) {
            // sleep(delay);
        }

        // Input split
        std::map<int, std::string> records = input_split(task.first);

        // Map
        std::map<std::string, int> intermediate_result;
        for(const auto& record : records) {
            std::map<std::string, int> result = map(record);
            for(const auto& pair : result) {
                intermediate_result[pair.first] += pair.second;
            }
        }

        // Output
        std::ofstream fout(output_dir + job_name + "-ir-" + std::to_string(task.first) + ".txt");
        for(const auto& pair : intermediate_result) {
            fout << pair.first << " " << pair.second << std::endl;
        }
        fout.close();

        // Decrement num_working
        pthread_mutex_lock(&mutex2);
        num_working--;
        pthread_cond_signal(&cond2); // Wake up main thread
        pthread_mutex_unlock(&mutex2);
    }
}

std::pair<int, int> TaskTracker::get_task() {
    // Send node_id to job tracker to request a map task using tag[0]
    MPI_Send(&node_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    // std::cout << "TaskTracker[" << node_id << "] MPI_Send to JobTracker requests map task" << std::endl;

    // Receive chunk_id & remote from job tracker using tag[1]
    int buffer[2] = {0, 0};
    MPI_Recv(buffer, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::cout << "TaskTracker[" << node_id << "] MPI_Recv from JobTracker chunk_id = " << buffer[0] << ", remote = " << buffer[1] << std::endl;

    // Return chunk_id
    return std::make_pair(buffer[0], buffer[1]);
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

void TaskTracker::req_reduce_tasks() {
    while(true) {
        // Send node_id to job tracker to request a reduce task using tag[0]
        MPI_Send(&node_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        // Receive job_id from job tracker using tag[1]
        int job_id = 0;
        MPI_Recv(&job_id, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // If job_id == -1, exit
        if(job_id == -1) {
            break;
        }

        // Read shuffle file
        std::ifstream fin(output_dir + job_name + "-shuffle-" + std::to_string(job_id) + ".txt");
        std::string line;
        std::vector<std::pair<std::string, int>> data;
        while(std::getline(fin, line)) {
            std::stringstream ss(line);
            std::string word;
            int count;
            ss >> word >> count;
            data.push_back(std::make_pair(word, count));
        }

        // Sort by keys (default)
        data = sort(data);

        // Group by exact same key (default)
        std::map<std::string, std::vector<int>> grouped_data = group(data);

        // Reduce
        std::vector<std::pair<std::string, int>> reduced_data;
        for(const auto& pair : grouped_data) {
            reduced_data.push_back(reduce(pair));
        }

        // Output
        output(reduced_data, job_id);
    }
}

std::vector<std::pair<std::string, int>> TaskTracker::sort(std::vector<std::pair<std::string, int>> data) {
    // Sort by keys (default)
    std::sort(data.begin(), data.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    return data;
}

std::map<std::string, std::vector<int>> TaskTracker::group(std::vector<std::pair<std::string, int>> data) {
    // Group by exact same key (default)
    std::map<std::string, std::vector<int>> grouped_data;
    for(const auto& pair : data) {
        grouped_data[pair.first].push_back(pair.second);
    }
    return grouped_data;
}

std::pair<std::string, int> TaskTracker::reduce(std::pair<std::string, std::vector<int>> pair) {
    int count = 0;
    for(const auto& c : pair.second) {
        count += c;
    }
    return std::make_pair(pair.first, count);
}

void TaskTracker::output(std::vector<std::pair<std::string, int>> data, int job_id) {
    // Output to file
    std::ofstream fout(output_dir + job_name + "-" + std::to_string(job_id) + ".out");
    for(const auto& pair : data) {
        fout << pair.first << " " << pair.second << std::endl;
    }
    fout.close();
}

// Utils
void TaskTracker::print() {
    std::cout << "[TaskTracker]" << std::endl;
    std::cout << "num_cpus: " << num_cpus << std::endl;
    std::cout << "node_id: " << node_id << std::endl;
    std::cout << "chunk_size: " << chunk_size << std::endl;
    std::cout << "word_file: " << word_file << std::endl;
}