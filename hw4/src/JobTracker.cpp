#include "JobTracker.hpp"

/* Constructor */
JobTracker::JobTracker(int num_nodes, std::string loc_config_file) {
    this->num_nodes = num_nodes;
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
        // std::cout << "line = " << line << std::endl;
        // std::cout << "chunk_id = " << chunk_id << ", node_id = " << node_id << std::endl;
    }
}

void JobTracker::dispatch_map_tasks() {
    // While there are still chunks
    while(loc_config.size() > 0) {
        // Recv node_id from task tracker using tag[0]
        int node_id;
        MPI_Recv(&node_id, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // std::cout << "JobTracker MPI_Recv from TaskTracker[" << node_id << "] requests map task" << std::endl;

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
        // std::cout << "JobTracker MPI_Send to TaskTracker[" << node_id << "] chunk_id = " << chunk_id << ", remote = " << remote << std::endl;

        // Remove chunk_id from loc_config
        loc_config.erase(chunk_id);
    }

    // Send -1 to all task trackers
    for(int i = 1; i < num_nodes; i++) {
        int node_id;
        MPI_Recv(&node_id, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int buffer[2] = {-1, 0};
        MPI_Send(buffer, 2, MPI_INT, i, 1, MPI_COMM_WORLD);
        std::cout << "JobTracker MPI_Send to TaskTracker[" << i << "] chunk_id = -1, remote = 0" << std::endl;
    }
    std::cout << "JobTracker dispatch_map_tasks() done" << std::endl;
}

/* Utils */
void JobTracker::print_loc_config() {
    for (const auto& p : loc_config) {
        std::cout << p.first << " " << p.second << std::endl;
    }
}
