#include <bits/stdc++.h>

#ifndef TASKTRACKER_HPP
#define TASKTRACKER_HPP

class TaskTracker {
public:
    // Variables
    int num_cpus;
    int node_id;
    int chunk_size;
    std::string word_file;

    // Constructor
    TaskTracker(int node_id, int chunk_size, std::string word_file);

    // Methods
    void set_num_cpus();

    // Utils
    void print();
};

#endif