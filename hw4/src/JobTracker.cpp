#include "JobTracker.hpp"

/* Constructor */
JobTracker::JobTracker(int num_nodes, int num_reducers, std::string job_name, std::string loc_config_file, std::string output_dir) {
    // Start timer
    clock_gettime(CLOCK_MONOTONIC, &prog_start_time);

    // Initialize variables
    this->num_nodes = num_nodes;
    this->num_reducers = num_reducers;
    this->job_name = job_name;
    this->output_dir = output_dir;
    set_loc_config(loc_config_file);
}

/* Methods */ 
void JobTracker::set_loc_config(std::string loc_config_file) {
    std::ifstream fin(loc_config_file);
    std::string line;
    while (std::getline(fin, line)) {
        std::stringstream ss(line);
        int chunk_id, node_id;
        ss >> chunk_id >> node_id;
        node_id = (node_id % (num_nodes - 1)) + 1;
        loc_config[chunk_id] = node_id;
    }
    num_chunks = loc_config.size();
}

void JobTracker::dispatch_map_tasks() {
    // While there are still chunks
    while(loc_config.size() > 0) {
        // Recv node_id from task tracker using tag[0]
        int node_id;
        MPI_Recv(&node_id, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "JobTracker Recv node " << node_id << std::endl;

        // Locality aware scheduling
        int chunk_id = -1;
        int remote = 1;
        for(const auto& p : loc_config) {
            if(p.second == node_id) {
                chunk_id = p.first;
                remote = 0;
                break;
            }
        }
        if(remote) {
            chunk_id = loc_config.begin()->first;
        }

        // Send chunk_id & remote to task tracker using tag[1]
        int buffer[2] = {chunk_id, remote};
        MPI_Send(buffer, 2, MPI_INT, node_id, 1, MPI_COMM_WORLD);
        std::cout << "JobTracker Send {" << buffer[0] << ", " << buffer[1] << "} to node " << node_id << std::endl;

        // Remove chunk_id from loc_config
        loc_config.erase(chunk_id);

        // Log
        log("Dispatch_MapTask," + std::to_string(chunk_id) + "," + std::to_string(node_id));
    }

    // Send -1 to all task trackers
    for(int i = 1; i < num_nodes; i++) {
        int node_id;
        MPI_Recv(&node_id, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "JobTracker Recv node " << node_id << " using tag 0" << std::endl;
        int buffer[2] = {-1, 0};
        MPI_Send(buffer, 2, MPI_INT, node_id, 1, MPI_COMM_WORLD);
        std::cout << "JobTracker Send {" << buffer[0] << ", " << buffer[1] << "} to node " << node_id << " using tag 1" << std::endl;
    }

    // Recv all tasks times using tag 3
    std::cout << "JobTracker Recv all tasks times using tag 3" << std::endl;
    for(int i = 0; i < num_chunks; i++) {
        int node_id;
        MPI_Recv(&node_id, 1, MPI_INT, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int chunk_id;
        MPI_Recv(&chunk_id, 1, MPI_INT, node_id, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double time;
        MPI_Recv(&time, 1, MPI_DOUBLE, node_id, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Log
        log("Complete_MapTask," + std::to_string(chunk_id) + "," + std::to_string(time));
    }
}

void JobTracker::shuffle() {
    // Start timer
    time_t start_t = time(nullptr);
    struct timespec shuffle_start_time, shuffle_end_time;
    clock_gettime(CLOCK_MONOTONIC, &shuffle_start_time);

    // Create num_reducers files
    std::vector<std::ofstream> fout(num_reducers);
    for(int i = 0; i < num_reducers; i++) {
        fout[i].open(output_dir + job_name + "-shuffle-" + std::to_string(i) + ".txt");
    }

    // Read from all intermediate files
    int pair_count = 0;
    for(int i = 1; i <= num_chunks; i++) {
        std::ifstream fin(output_dir + job_name + "-ir-" + std::to_string(i) + ".txt");
        std::string line;
        while (std::getline(fin, line)) {
            std::stringstream ss(line);
            std::string word;
            int count;
            ss >> word >> count;
            int offset = partition(word);
            fout[offset] << word << " " << count << std::endl;
            pair_count++;
        }
    }

    // Close all files
    for(int i = 0; i < num_reducers; i++) {
        fout[i].close();
    }

    // Log
    clock_gettime(CLOCK_MONOTONIC, &shuffle_end_time);
    double elapsed_sec = calc_time(shuffle_start_time, shuffle_end_time);
    log(std::to_string(start_t) + ",Start_Shuffle," + std::to_string(pair_count), false);
    log("Finish_Shuffle," + std::to_string(elapsed_sec));
}

int JobTracker::partition(std::string key) {
    int offset = key[0] - 'A';
    return offset % num_reducers;
}

void JobTracker::dispatch_reduce_tasks() {
    // While there are still shuffle files
    for(int i = 0; i < num_reducers; i++) {
        // Recv node_id from task tracker using tag[7]
        int node_id;
        MPI_Recv(&node_id, 1, MPI_INT, MPI_ANY_SOURCE, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Send shuffle file id to task tracker using tag[8]
        MPI_Send(&i, 1, MPI_INT, node_id, 8, MPI_COMM_WORLD);

        // Log
        log("Dispatch_ReduceTask," + std::to_string(i) + "," + std::to_string(node_id));
    }

    // Send -1 to all task trackers
    for(int i = 1; i < num_nodes; i++) {
        int node_id;
        MPI_Recv(&node_id, 1, MPI_INT, i, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int buffer = -1;
        MPI_Send(&buffer, 1, MPI_INT, i, 8, MPI_COMM_WORLD);
        std::cout << "JobTracker MPI_Send to TaskTracker[" << i << "] shuffle file id = -1" << std::endl;
    }
    std::cout << "JobTracker dispatch_reduce_tasks() done" << std::endl;

    // Recv all tasks times using tag 4
    for(int i = 0; i < num_reducers; i++) {
        int node_id;
        MPI_Recv(&node_id, 1, MPI_INT, MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int reduce_id;
        MPI_Recv(&reduce_id, 1, MPI_INT, node_id, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double time;
        MPI_Recv(&time, 1, MPI_DOUBLE, node_id, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Log
        log("Complete_ReduceTask," + std::to_string(reduce_id) + "," + std::to_string(time));
    }
}

void JobTracker::log(std::string str, bool keep_time) {
    std::ofstream fout(output_dir + job_name + "-log.out", std::ios::app);
    assert(fout.is_open());
    if(keep_time)
        fout << time(nullptr) << "," << str << std::endl;
    else
        fout << str << std::endl;
    fout.close();
}

double JobTracker::calc_time(struct timespec start, struct timespec end) {
    double elapsed_sec = end.tv_sec - start.tv_sec;
    double elapsed_nsec = end.tv_nsec - start.tv_nsec;
    return elapsed_sec + elapsed_nsec / 1000000000.0;
}

void JobTracker::finish() {
    clock_gettime(CLOCK_MONOTONIC, &prog_end_time);
    double elapsed = calc_time(prog_start_time, prog_end_time);
    log("Finish_Job," + std::to_string(elapsed));
}

/* Utils */
void JobTracker::print_loc_config() {
    for (const auto& p : loc_config) {
        std::cout << p.first << " " << p.second << std::endl;
    }
}

void JobTracker::verify_ir() {
    std::map<std::string, int> ans;
    for(int i = 1; i <= num_chunks; i++) {
        std::ifstream fin(output_dir + job_name + "-ir-" + std::to_string(i) + ".txt");
        std::string line;
        while (std::getline(fin, line)) {
            std::stringstream ss(line);
            std::string word;
            int count;
            ss >> word >> count;
            ans[word] += count;
        }
    }
    std::ofstream fout(output_dir + job_name + "-ir-ans.txt");
    for(const auto& p : ans) {
        fout << p.first << " " << p.second << std::endl;
    }
}

void JobTracker::verify_shuffle() {
    std::map<std::string, int> ans;
    for(int i = 0; i < num_reducers; i++) {
        std::ifstream fin(output_dir + job_name + "-shuffle-" + std::to_string(i) + ".txt");
        std::string line;
        while (std::getline(fin, line)) {
            std::stringstream ss(line);
            std::string word;
            int count;
            ss >> word >> count;
            ans[word] += count;
        }
    }
    std::ofstream fout(output_dir + job_name + "-shuffle-ans.txt");
    for(const auto& p : ans) {
        fout << p.first << " " << p.second << std::endl;
    }
}

void JobTracker::verify_reduce() {
    std::map<std::string, int> ans;
    for(int i = 0; i < num_reducers; i++) {
        std::ifstream fin(output_dir + job_name + "-" + std::to_string(i) + ".out");
        std::string line;
        while (std::getline(fin, line)) {
            std::stringstream ss(line);
            std::string word;
            int count;
            ss >> word >> count;
            ans[word] += count;
        }
    }
    std::ofstream fout(output_dir + job_name + "-reduce-ans.out");
    for(const auto& p : ans) {
        fout << p.first << " " << p.second << std::endl;
    }
}