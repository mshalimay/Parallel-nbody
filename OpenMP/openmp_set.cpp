#include "common.h"
#include <omp.h>
#include <cmath>
#include <queue>
#include <iostream>

// Put any static global variables here that you will use throughout the simulation.
// Global grid instance
grid_thread_safe grid_omp;

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    
    // if distance two particles > cutoff, no force is applied = don't interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    
    p.vx += p.ax * dt; // velocity += acceleration * time
    p.vy += p.ay * dt;
    p.x += p.vx * dt;  // position += velocity * time
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

// thread safe grid update blocks method
void grid_thread_safe::update_blocks(particle_t* parts, int num_parts) {
    #pragma omp parallel for
    for (int i = 0; i < num_parts; ++i) {
        // update particle's block
        int block_id = grid_omp.get_block_index(parts[i].x, parts[i].y);
        if (parts[i].block_id != block_id) {
            // remove particle from old block
            grid_omp.blocks[parts[i].block_id].remove_particle(i);
            // add particle to new block and update particle's block_id
            grid_omp.blocks[block_id].add_particle(i);
            // update particle's block_id
            parts[i].block_id = block_id;
        }
        // reset acceleration
        parts[i].ax = parts[i].ay = 0;
    }
}

void move_update_reset(int num_parts, particle_t* parts, double size) {
    //  Move Particles
    #pragma omp parallel for
    for (int i = 0; i < num_parts; ++i) {
        // move particle
        move(parts[i], size);

        // update particle's block
        int block_id = grid_omp.get_block_index(parts[i].x, parts[i].y);
        if (parts[i].block_id != block_id) {
            // remove particle from old block
            grid_omp.blocks[parts[i].block_id].remove_particle(i);
            // add particle to new block and update particle's block_id
            grid_omp.blocks[block_id].add_particle(i);
            // update particle's block_id
            parts[i].block_id = block_id;
        }
        // reset acceleration
        parts[i].ax = parts[i].ay = 0;
    }
}

void apply_force_blocks(int block1, int block2, particle_t* parts) {
    for (int i : grid_omp.blocks[block1].particles) {
        for (int j : grid_omp.blocks[block2].particles) {
            if (i==j) continue;
            apply_force(parts[i], parts[j]);
        }
    }
}


//SECTION: VERSION 1
//============================================================================
// Version 1 - each block has one task queue
//============================================================================

// tuple of vectors of queues
std::tuple<std::vector<std::queue<int>>, std::vector<std::queue<int>>> task_queues = 
    std::make_tuple(std::vector<std::queue<int>>(), std::vector<std::queue<int>>());
int current_idx = 0;


void populate_tasks (std::vector<std::queue<int>> &tasks){
    for (int block_id = 0; block_id < grid_omp.blocks.size(); ++block_id) {    
        for (int drow = -1; drow <= 1; drow++) {
            for (int dcol = -1; dcol <= 1; dcol++) {
                int neighbor_block_row = (block_id / grid_omp.blocks_per_row) + drow;
                int neighbor_block_col = (block_id % grid_omp.blocks_per_row) + dcol;

                // Check if the neighbor block is within bounds
                if (neighbor_block_row >= 0 && neighbor_block_row < grid_omp.blocks_per_row &&
                    neighbor_block_col >= 0 && neighbor_block_col < grid_omp.blocks_per_row) {                    
                    // if neighbor block within bounds, add to list
                    tasks[block_id].push(neighbor_block_row * grid_omp.blocks_per_row + neighbor_block_col);
                }
            }
        }
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    grid_omp.block_size = bsize * cutoff;
    grid_omp.blocks_per_row = int(ceil(size / grid_omp.block_size));
    int total_blocks = grid_omp.blocks_per_row * grid_omp.blocks_per_row;

    // initialize (blocks_per_row x blocks_per_row) grid of blocks representing the xy plane
    grid_omp.blocks.reserve(total_blocks);

    // initialize vectors of queues
    std::get<0>(task_queues).resize(total_blocks);
    std::get<1>(task_queues).resize(total_blocks);

    // create a vector of blocks
    for (int i = 0; i < total_blocks; ++i) {
        grid_omp.blocks.push_back(block_thread_safe());
    }

    // Assign particles to respective blocks based on `x` and `y` coordinates
    for (int i = 0; i < num_parts; ++i) {
        int block_id = grid_omp.get_block_index(parts[i].x, parts[i].y);
        grid_omp.blocks[block_id].add_particle_unsafe(i);
        parts[i].block_id = block_id;
    }

    // create a list of tasks for each block
    auto &tasks_list = (current_idx == 0 ? std::get<0>(task_queues) : std::get<1>(task_queues));

    populate_tasks(tasks_list);

    // init accelerations to zero for all particles
    for (int i = 0; i < num_parts; ++i) {
        parts[i].ax = parts[i].ay = 0;
    }
}


// iterate over the blocks
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    std::vector<std::queue<int>> &task_list = (current_idx == 0 ? std::get<0>(task_queues) : std::get<1>(task_queues));
    std::vector<std::queue<int>> &next_task_list = (current_idx == 0 ? std::get<1>(task_queues) : std::get<0>(task_queues));
    
    // Iterate over each block
    #pragma omp parallel for
    for (int block_id = 0; block_id < grid_omp.blocks.size(); ++block_id) {
        // process the queue of tasks: try to acquire a lock on a pair of blocks;
        // if success, apply force in particles within blocks;  if not, re-enqueue the task
        
        // tasks queue
        std::queue<int>& tasks = task_list[block_id];
        std::queue<int>& next_tasks = next_task_list[block_id];

        while (!tasks.empty()) {
            // pop task from tasks
            int current_task = tasks.front();
            tasks.pop();

            // try to acquire lock on current block
            block_thread_safe& current_block = grid_omp.blocks[block_id];
            std::unique_lock<std::mutex> lock1(current_block.mtx, std::try_to_lock);
            
            if (not lock1.owns_lock()) {
                tasks.push(current_task);
                continue;
            }

            // if tasks is to apply to itself, need only one lock
            if (current_task == block_id and lock1.owns_lock()){
                apply_force_blocks(block_id, block_id, parts);
                next_tasks.push(current_task);
                continue;
            }
                  
            // try to acquire lock on neighbor block
            block_thread_safe& neighbor_block = grid_omp.blocks[current_task];
            std::unique_lock<std::mutex> lock2(neighbor_block.mtx, std::try_to_lock);

            if (lock1.owns_lock() and lock2.owns_lock()) {
                // Lock acquired, process the blocks
                apply_force_blocks(block_id, current_task, parts);    
                next_tasks.push(current_task);   
            } else {
                // Lock not acquired, re-enqueue the task and proceed to next
                tasks.push(current_task);
            }
        }
    }
    current_idx = 1 - current_idx;
    move_update_reset(num_parts, parts, size);

    // less performance
    // #pragma omp parallel for
    // for (int i = 0; i < num_parts; ++i) {
    //     // move particle
    //     move(parts[i], size);
    //     // reset acceleration
    //     parts[i].ax = parts[i].ay = 0;
    // }
    // // update particles' blocks
    // grid_omp.update_blocks(parts, num_parts);
}

//\SECTION

//SECTION: VERSION 2
//============================================================================
// Version 2 - common tasks with stealing
//============================================================================
// std::tuple<task_queue, task_queue> task_queues = std::make_tuple(task_queue(), task_queue());
// int queue_idx = 0;

// void populate_queue (task_queue &tasks){
//     for (int block_id = 0; block_id < grid_omp.blocks.size(); ++block_id) {
//         for (int drow = -1; drow <= 1; drow++) {
//             for (int dcol = -1; dcol <= 1; dcol++) {
//                 int neighbor_block_row = (block_id / grid_omp.blocks_per_row) + drow;
//                 int neighbor_block_col = (block_id % grid_omp.blocks_per_row) + dcol;

//                 // Check if the neighbor block is within bounds
//                 if (neighbor_block_row >= 0 && neighbor_block_row < grid_omp.blocks_per_row &&
//                     neighbor_block_col >= 0 && neighbor_block_col < grid_omp.blocks_per_row) {                    
//                     // if neighbor block within bounds, enqueue
//                     int neighbor_id = neighbor_block_row * grid_omp.blocks_per_row + neighbor_block_col;
//                     tasks.push_unsafe(std::make_tuple(block_id, neighbor_id));
//                 }
//             }
//         }
//     }
// }
    
// void init_simulation(particle_t* parts, int num_parts, double size, double bsize) {
//     grid_omp.block_size = bsize * cutoff;
//     grid_omp.blocks_per_row = int(ceil(size / grid_omp.block_size));

//     // initialize (blocks_per_row x blocks_per_row) grid of blocks representing the xy plane
//     grid_omp.blocks.reserve(grid_omp.blocks_per_row * grid_omp.blocks_per_row);

//     // create a vector of blocks
//     for (int i = 0; i < grid_omp.blocks_per_row * grid_omp.blocks_per_row; ++i) {
//         grid_omp.blocks.push_back(block_thread_safe());
//     }

//     // Assign particles to respective blocks based on `x` and `y` coordinates
//     for (int i = 0; i < num_parts; ++i) {
//         int block_id = grid_omp.get_block_index(parts[i].x, parts[i].y);
//         grid_omp.blocks[block_id].add_particle_unsafe(i);
//         parts[i].block_id = block_id;
//     }
//     // populate the current task queue
//     task_queue &tasks = (queue_idx == 0 ? std::get<0>(task_queues) : std::get<1>(task_queues));
//     populate_queue(tasks);

//     // Reset accelerations to zero for all particles
//     for (int i = 0; i < num_parts; ++i) {
//         parts[i].ax = parts[i].ay = 0;
//     }

// }


// // iterate over the blocks
// void simulate_one_step(particle_t* parts, int num_parts, double size) {

//     // get current task queue from tuple
//     task_queue &tasks = (queue_idx == 0 ? std::get<0>(task_queues) : std::get<1>(task_queues));
//     task_queue &next_tasks = (queue_idx == 0 ? std::get<1>(task_queues) : std::get<0>(task_queues));

//     // populate the task queue
//     // populate_queue(tasks, task_list);
//     #pragma omp parallel
//     {    
//         // print the thread id
//         // std::cout << "Thread " << omp_get_thread_num() << " is running" << std::endl;

//         std::pair<bool, std::tuple<int, int>> task;
//         int block_id, neighbor_id;
//         while (true){
//             task = tasks.pop();
            
//             if (not task.first) {
//                 // std::cout << "Thread " << omp_get_thread_num() << " finished" << std::endl;
//                 break;
//             }

//             block_id = std::get<0>(task.second);
//             neighbor_id = std::get<1>(task.second);

//             // try to acquire lock on current block
//             block_thread_safe& current_block = grid_omp.blocks[block_id];
//             std::unique_lock<std::mutex> lock1(current_block.mtx, std::try_to_lock);
            
//             if (not lock1.owns_lock()) {
//                 tasks.push(std::make_tuple(block_id, neighbor_id));
//                 continue;
//             }

//             // if tasks is to apply to itself, need only one lock
//             if (block_id == neighbor_id and lock1.owns_lock()){
//                 // std::cout << "Thread " << omp_get_thread_num() << " processing task" << block_id << " " << neighbor_id << std::endl;
//                 apply_force_blocks(block_id, block_id, parts);
//                 // push task to the next queue
//                 next_tasks.push(std::make_tuple(block_id, neighbor_id));
//                 continue;
//             }
                
//             // try to acquire lock on neighbor block
//             block_thread_safe& neighbor_block = grid_omp.blocks[neighbor_id];
//             std::unique_lock<std::mutex> lock2(neighbor_block.mtx, std::try_to_lock);

//             if (lock1.owns_lock() and lock2.owns_lock()) {
//                 // Lock acquired, process the blocks
//                 apply_force_blocks(block_id, neighbor_id, parts);
//                 // push task to the next queue
//                 next_tasks.push(std::make_tuple(block_id, neighbor_id));
//             } else {
//                 // Lock not acquired, re-enqueue the task and proceed to next
//                 tasks.push(std::make_tuple(block_id, neighbor_id));
//             }
//         }
//     }
//     queue_idx = 1 - queue_idx;

//     move_update_reset(num_parts, parts, size);
// }
// //\SECTION