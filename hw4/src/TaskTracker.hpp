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
    static int node_id;
    static int chunk_size;
    int num_reducers;
    static std::string job_name;
    static std::string word_file;
    static std::string output_dir;
    static int delay;
    static pthread_mutex_t mutex;
    static pthread_cond_t cond;
    static std::queue<std::pair<int, int>> tasks;
    static int num_working;
    static pthread_mutex_t mutex2;
    static pthread_cond_t cond2;
    static std::map<int, double> map_task_time;
    static std::map<int, double> reduce_task_time;

    /* Constructor */
    TaskTracker(int node_id, int chunk_size, int delay, int num_reducers, std::string job_name, std::string word_file, std::string output_dir);

    /* Methods */
    std::pair<int, int> get_task();
    void set_num_cpus();
    void req_map_tasks();
    static void* map_thread_func(void* thread_id);
    static double calc_time(struct timespec start, struct timespec end);
    static std::map<int, std::string> input_split(int chunk_id);
    static std::map<std::string, int> map(std::pair<int, std::string> record);
    void req_reduce_tasks();
    std::vector<std::pair<std::string, int>> sort(std::vector<std::pair<std::string, int>> pairs);
    std::map<std::string, std::vector<int>> group(std::vector<std::pair<std::string, int>> pairs);
    std::pair<std::string, int> reduce(std::pair<std::string, std::vector<int>> pair);
    void output(std::vector<std::pair<std::string, int>> pairs, int job_id);

    /* Utils */
    void print();
};

#endif