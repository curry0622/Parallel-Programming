#include <bits/stdc++.h>
#include <mpi.h>

#ifndef TASKTRACKER_HPP
#define TASKTRACKER_HPP

class TaskTracker {
public:
    // Variables
    int num_cpus;
    int node_id;
    int chunk_size;
    std::string word_file;
    std::map<int, std::string> records;

    // Constructor
    TaskTracker(int node_id, int chunk_size, std::string word_file);

    // Methods
    void set_num_cpus();
    void input_split(int chunk_id);
    void map(int line_id);
    int request_map_task();

    // Utils
    void print();
};

#endif