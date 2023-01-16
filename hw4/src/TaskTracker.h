#ifndef TASKTRACKER_HPP
#define TASKTRACKER_HPP

#include <bits/stdc++.h>
#include <mpi.h>
#include <pthread.h>
#include <unistd.h>

class TaskTracker {
public:
    /* Variables */
    int num_cpus;
    inline static int node_id;
    inline static int chunk_size;
    int num_reducers;
    inline static std::string job_name;
    inline static std::string word_file;
    inline static std::string output_dir;
    inline static int delay;
    inline static pthread_mutex_t mutex;
    inline static pthread_cond_t cond;
    inline static std::queue<std::pair<int, int>> tasks;
    inline static int num_working;
    inline static pthread_mutex_t mutex2;
    inline static pthread_cond_t cond2;
    inline static std::map<int, double> map_task_time;
    inline static std::map<int, double> reduce_task_time;

    /* Constructor */
    TaskTracker(int node_id, int chunk_size, int delay, int num_reducers, std::string job_name, std::string word_file, std::string output_dir) {
        this->node_id = node_id;
        this->chunk_size = chunk_size;
        this->delay = delay;
        this->num_reducers = num_reducers;
        this->job_name = job_name;
        this->word_file = word_file;
        this->output_dir = output_dir;
        set_num_cpus();
    }

    /* Methods */
    std::pair<int, int> get_task() {
        // Send node_id to job tracker to request a map task using tag[0]
        MPI_Send(&node_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        std::cout << "TaskTracker[" << node_id << "] MPI_Send to JobTracker requests map task" << std::endl;

        // Receive chunk_id & remote from job tracker using tag[1]
        int buffer[2] = {0, 0};
        std::cout << "TaskTracker[" << node_id << "] waiting msg from JobTracker" << std::endl;
        MPI_Recv(buffer, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "TaskTracker[" << node_id << "] MPI_Recv from JobTracker chunk_id = " << buffer[0] << ", remote = " << buffer[1] << std::endl;

        // Return chunk_id
        return std::make_pair(buffer[0], buffer[1]);
    }

    void set_num_cpus() {
        cpu_set_t cpu_set;
        sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
        num_cpus = CPU_COUNT(&cpu_set);
    }

    void req_map_tasks() {
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
            std::cout << "TaskTracker " << node_id << " got task " << task.first << " " << task.second << std::endl;

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

        // Send task times to job tracker using tag 3
        for(const auto& pair : map_task_time) {
            MPI_Send(&node_id, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
            MPI_Send(&pair.first, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
            MPI_Send(&pair.second, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
        }
    }

    static void* map_thread_func(void* thread_id) {
        // Timer
        struct timespec task_start_time, task_end_time;

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

            // Start timer
            clock_gettime(CLOCK_MONOTONIC, &task_start_time);

            // If task is remote, sleep
            if(task.second) {
                std::cout << "TaskTracker " << node_id << ", Thread " << *(int*)thread_id << ": Sleeping for remote task " << task.first << std::endl;
                sleep(delay);
                std::cout << "TaskTracker " << node_id << ", Thread " << *(int*)thread_id << ": Woke up for remote task " << task.first << std::endl;
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

            // End timer
            clock_gettime(CLOCK_MONOTONIC, &task_end_time);
            double task_elapsed_sec = calc_time(task_start_time, task_end_time);

            // Decrement num_working
            pthread_mutex_lock(&mutex2);
            map_task_time[task.first] = task_elapsed_sec;
            num_working--;
            pthread_cond_signal(&cond2); // Wake up main thread
            pthread_mutex_unlock(&mutex2);
        }
    }

    static double calc_time(struct timespec start, struct timespec end) {
        double elapsed_sec = end.tv_sec - start.tv_sec;
        double elapsed_nsec = end.tv_nsec - start.tv_nsec;
        return elapsed_sec + elapsed_nsec / 1000000000.0;
    }

    static std::map<int, std::string> input_split(int chunk_id) {
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

    static std::map<std::string, int> map(std::pair<int, std::string> record) {
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

    void req_reduce_tasks() {
        // Timer
        struct timespec task_start_time, task_end_time;

        while(true) {
            // Send node_id to job tracker to request a reduce task using tag[7]
            MPI_Send(&node_id, 1, MPI_INT, 0, 7, MPI_COMM_WORLD);

            // Receive job_id from job tracker using tag[8]
            int job_id = 0;
            MPI_Recv(&job_id, 1, MPI_INT, 0, 8, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // If job_id == -1, exit
            if(job_id == -1) {
                break;
            }

            // Start timer
            clock_gettime(CLOCK_MONOTONIC, &task_start_time);

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
            // std::map<std::string, std::vector<int>> grouped_data = group(data);
            std::vector<std::pair<std::string, std::vector<int>>> grouped_data = group(data);

            // Reduce
            std::vector<std::pair<std::string, int>> reduced_data;
            for(const auto& pair : grouped_data) {
                reduced_data.push_back(reduce(pair));
            }

            // Output
            output(reduced_data, job_id);

            // End timer
            clock_gettime(CLOCK_MONOTONIC, &task_end_time);
            double elapsed_time = calc_time(task_start_time, task_end_time);
            reduce_task_time[job_id] = elapsed_time;
        }

        // Send time to job tracker using tag 4
        for(const auto& pair : reduce_task_time) {
            int job_id = pair.first;
            double time = pair.second;
            MPI_Send(&node_id, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
            MPI_Send(&job_id, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
            MPI_Send(&time, 1, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
        }
    }

    std::vector<std::pair<std::string, int>> sort(std::vector<std::pair<std::string, int>> pairs) {
        // Sort by keys in ascending order (default)
        std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        // Sort by keys in descending order (demo)
        // std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
        //     return a.first > b.first;
        // });
        return pairs;
    }

    std::vector<std::pair<std::string, std::vector<int>>> group(std::vector<std::pair<std::string, int>> pairs) {
        std::vector<std::pair<std::string, std::vector<int>>> grouped_data;
        // Group by exact same key (default)
        for(const auto& pair : pairs) {
            if(grouped_data.empty() || grouped_data.back().first != pair.first) {
                grouped_data.push_back(std::make_pair(pair.first, std::vector<int>()));
            }
            grouped_data.back().second.push_back(pair.second);
        }
        // Group by first character (demo)
        // for(const auto& pair : pairs) {
        //     if(grouped_data.empty() || grouped_data.back().first[0] != pair.first[0]) {
        //         grouped_data.push_back(std::make_pair(pair.first.substr(0, 1), std::vector<int>()));
        //     }
        //     grouped_data.back().second.push_back(pair.second);
        // }
        return grouped_data;
    }

    std::pair<std::string, int> reduce(std::pair<std::string, std::vector<int>> pair) {
        int count = 0;
        for(const auto& c : pair.second) {
            count += c;
        }
        return std::make_pair(pair.first, count);
    }

    void output(std::vector<std::pair<std::string, int>> pairs, int job_id) {
        // Output to file
        std::ofstream fout(output_dir + job_name + "-" + std::to_string(job_id) + ".out");
        for(const auto& pair : pairs) {
            fout << pair.first << " " << pair.second << std::endl;
        }
        fout.close();
    }

    /* Utils */
};

#endif