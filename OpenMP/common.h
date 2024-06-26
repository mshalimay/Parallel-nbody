#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

// Program Constants
#define nsteps   1000
#define savefreq 10
#define density  0.0005
#define mass     0.01
#define cutoff   0.01   // if distance two particles < cutoff, they interact
#define min_r    (cutoff / 100)
#define dt      0.0005

#include <vector>
#include <unordered_set>
#include <pthread.h>
#include <mutex>
#include <optional>
#include <queue>
#include <unordered_map>

// Particle Data Structure
typedef struct particle_t {
    double x;     // Position X
    double y;     // Position Y
    double vx;    // Velocity X
    double vy;    // Velocity Y
    double ax;    // Acceleration X
    double ay;    // Acceleration Y
    int block_id; // block the particle belongs to
    // int index_in_block = -1; // could use this attribute, but to make parallel and sequential versions more similar, we will not use it
} particle_t;

struct block {    
    // Contains the indices of particles
    std::vector<int> particles;
    // maps a particle to its position within the `particles` vector 
    std::unordered_map<int, int> part_idxs;
    
    block() = default;

    block(block&& other) noexcept {
        particles = std::move(other.particles);  
        part_idxs = std::move(other.part_idxs);  
    }

    // remove particle thread-safely
    void remove_particle(int particle_index) {
        // get the index of the particle within the block
        int idx_within_block = part_idxs[particle_index];
        if (idx_within_block < particles.size() - 1) {
            // swap the particle with the last particle in the block
            std::swap(particles[idx_within_block], particles.back());
            // update the index of the swapped particle
            part_idxs[particles[idx_within_block]] = idx_within_block;
        }
        // remove the particle from the block (vector and and index map)
        particles.pop_back();
        part_idxs.erase(particle_index);
    }

    void add_particle(int particle_index) {
        part_idxs[particle_index] = particles.size();
        particles.push_back(particle_index);    
    }    
};


struct grid {
    // grid is an abstract representation of 2D xy space divided into square blocks
    std::vector<block> blocks;

    int blocks_per_row; // Number of blocks along one dimension
    double block_size;  // length of the block's side

    // Calculate which cell a particle belongs to based on its position
    int get_block_index(double x, double y) const {
        // eg: block_size = 4, 3 blocks per row => xy grid from 0 to 11 is divided into 3x3 squares of size 4
          // in memory: [0,1,2] 1st row [3,4,5] 2nd row [6,7,8] 3rd row
          // xy coordinate: p = (11,5)

        // column of the block = length in horizontal direction (x) / block_size
          // eg: 5 / 4 = 1 => block is located in the second column in the 3x3 grid of blocks
        int j = int(x / block_size);
        
        // row index of the block = length in vertical direction (y) / block_size
          // eg: 11 / 4 = 2 => block is located in the uppmost row in the 3x3 grid of blocks
        int i = int(y / block_size);

        // return the index of the block in the grid converting (i,j) to row-major order
         // eg: 2 * 3 = 6 (walks to the 3rd row) + 1 (walks to the 2nd column in the 3rd row) = 7 
        return i * blocks_per_row + j;
    }

    //@i is the index of the particle whose block is to be updated
    // swap and pop + vector approach
    void update_particle_block(int i, particle_t* parts) {
        int block_id = get_block_index(parts[i].x, parts[i].y);
        
        // if particle did not change blocks, return
        if (parts[i].block_id == block_id){
            return;
        }
        
        // remove particle from old block
        blocks[parts[i].block_id].remove_particle(i);

        // add particle to new block and update its block id
        blocks[block_id].add_particle(i);
        parts[i].block_id = block_id;
    }

    // swap and pop + vector approach
    void update_blocks(particle_t* parts, int num_parts) {
        for (int i = 0; i < num_parts; ++i) {
            update_particle_block(i, parts);
        }
    }
};

//============================================================================
// thread-safe data structures
//============================================================================

// Thread-safe block structure
struct block_thread_safe {    
    // Contains the indices of particles
    std::vector<int> particles;
    // maps particle to its position within the `particles` vector 
    std::unordered_map<int, int> part_idxs;
    // std::unordered_set<int> particles;
    
    // Mutex for this block
    std::mutex mtx;                    

    block_thread_safe() = default;

    block_thread_safe(block_thread_safe&& other) noexcept {
        particles = std::move(other.particles);  
        part_idxs = std::move(other.part_idxs);  
    }

    // remove particle thread-safely
    void remove_particle(int particle_index) {
        std::lock_guard<std::mutex> guard(mtx);
        // get the index of the particle within the block
        int idx_within_block = part_idxs[particle_index];

        if (idx_within_block < particles.size() - 1) {
            // swap the particle with the last particle in the block
            std::swap(particles[idx_within_block], particles.back());
            // update the index of the swapped particle
            part_idxs[particles[idx_within_block]] = idx_within_block;
        }
        // remove the particle from the block (vector and and index map)
        particles.pop_back();
        part_idxs.erase(particle_index);
    }

    void add_particle(int particle_index) {
        std::lock_guard<std::mutex> guard(mtx);
        part_idxs[particle_index] = particles.size();
        particles.push_back(particle_index);    
    }

    // add particle thread-safely
    void add_particle_unsafe(int particle_index) {
        // insert particle into the index map
        part_idxs[particle_index] = particles.size();
        particles.push_back(particle_index);
    }
    
};

struct grid_thread_safe {
    std::vector<block_thread_safe> blocks;

    int blocks_per_row; 
    double block_size;  

    // Calculate which cell a particle belongs to based on its position
    int get_block_index(double x, double y) const {
        int j = int(x / block_size);
        int i = int(y / block_size);
        return i * blocks_per_row + j;
    }

    // void update_blocks(particle_t* parts, int num_parts);
};

// for the task stealing implementation (openmp.cpp version 2) only
struct task_queue{
    std::queue<std::tuple<int, int>> tasks;
    std::mutex mtx;

    task_queue() = default;

    task_queue(task_queue&& other) noexcept {
        tasks = std::move(other.tasks);  // Move the particles set
    }

    void push_unsafe(std::tuple<int, int> task) {
        tasks.push(task);
    }

    void push(std::tuple<int, int> task) {
        std::lock_guard<std::mutex> guard(mtx);
        tasks.push(task);
    }

    // std::optional<std::tuple<int, int>> pop() 
    std::pair<bool, std::tuple<int, int>> pop(){
        std::lock_guard<std::mutex> guard(mtx);

        if (!tasks.empty()) {
            auto task = tasks.front(); 
            tasks.pop(); 
            return std::make_pair(true, task); 
        }
        // Return failure
        return std::make_pair(false, std::tuple<int, int>());
    }

    bool empty() {
        std::lock_guard<std::mutex> guard(mtx);
        return tasks.empty();
    }
};


//============================================================================
// function prototypes and extern
//============================================================================

extern grid simulation_grid;

// Simulation routine
void init_simulation(particle_t* parts, int num_parts, double size);
void simulate_one_step(particle_t* parts, int num_parts, double size);

void move_update_reset(int num_parts, particle_t* parts, double size);

#endif
