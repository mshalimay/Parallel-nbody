#include "common.h"
#include <mpi.h>
#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <map>

// Static variables
struct block;
struct grid;

struct grid {
    // grid is an abstract representation of 2D xy space divided into square blocks
    std::vector<block> blocks;

    int blocks_per_row; // Number of blocks along one dimension
    double block_size;  // length of the block's side
    int total_blocks;   // total number of blocks

    // Calculate which block a particle belongs to based on its position
    int get_block_index(double x, double y) const {
        int j = int(x / block_size);        
        int i = int(y / block_size);
        return i * blocks_per_row + j;
    }
};

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

    void remove_particle(int particle_index) {
        int idx_within_block = part_idxs[particle_index];
        if (idx_within_block < particles.size() - 1) {
            std::swap(particles[idx_within_block], particles.back());
            part_idxs[particles[idx_within_block]] = idx_within_block;
        }
        // remove the particle from the block (vector and index map)
        particles.pop_back();
        part_idxs.erase(particle_index);
    }

    void add_particle(int particle_index) {
        part_idxs[particle_index] = particles.size();
        particles.push_back(particle_index);    
    }    
};


// Static global variables
int block_size = 6;
int start_block;
int end_block;
grid global_grid;
int total_blocks;
std::vector<block> blocks;
std::unordered_map<int, int> block_ids;
std::vector<std::vector<block*>> tasks;
std::unordered_map<int, int> part_to_block;



// function declarations
void update_blocks(particle_t* parts, int num_parts);
void update_particle_block(int i, particle_t* parts);
void create_blocks_and_tasks(int start_block, int end_block);
void apply_force(particle_t& particle, particle_t& neighbor);
void move(particle_t& p, double size);


// swap and pop + vector approach
void update_blocks(particle_t* parts, int num_parts) {
    for (int i = 0; i < num_parts; ++i) {
        update_particle_block(i, parts);
    }
}

void update_particle_block(int i, particle_t* parts) {
    int block_id = global_grid.get_block_index(parts[i].x, parts[i].y);

    // if particle did not change blocks, return
    if (parts[i].id == block_id){
        return;
    }

    // remove particle from old block
    blocks[parts[i].id].remove_particle(i);

    // add particle to new block and update its block id
    blocks[block_id].add_particle(i);
    parts[i].id = block_id;
}


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


// create the blocks and tasks for each block
void create_blocks(int start_block, int end_block) {
    // Initialize or clear existing data structures
    blocks.clear();
    block_ids.clear();

    int id = 0; // Local ID for indexing within 'blocks' and 'tasks'
    for (int block_id = start_block; block_id < end_block; ++block_id) {
        blocks.push_back(block()); // Create a new block
        block_ids[block_id] = id; // Map global block ID to local index
        tasks.push_back(std::vector<block*>()); // Initialize task list for this block
        
        int local_block_id = id++; // Use 'id' then increment for the next block
        // Calculate neighbors
        for (int drow = -1; drow <= 1; drow++) {
            for (int dcol = -1; dcol <= 1; dcol++) {
                // Calculate global IDs of neighbor blocks
                int neighbor_row = (block_id / global_grid.blocks_per_row) + drow;
                int neighbor_col = (block_id % global_grid.blocks_per_row) + dcol;

                if (neighbor_row >= 0 && neighbor_row < global_grid.blocks_per_row &&
                    neighbor_col >= 0 && neighbor_col < global_grid.blocks_per_row) {
                    int neighbor_id = neighbor_row * global_grid.blocks_per_row + neighbor_col;

                    if (block_ids.find(neighbor_id) == block_ids.end()) {
                        blocks.push_back(block());
                        block_ids[neighbor_id] = id;
                        id++;
                    }
                }
            }
        }
    }
}


// void populate_block_tasks(block_id){
//     // Add the current block to the task list of all its neighbors
//     for (int drow = -1; drow <= 1; drow++) {
//         for (int dcol = -1; dcol <= 1; dcol++) {
//             // Calculate global IDs of neighbor blocks
//             int neighbor_row = (block_id / global_grid.blocks_per_row) + drow;
//             int neighbor_col = (block_id % global_grid.blocks_per_row) + dcol;

//             if (neighbor_row >= 0 && neighbor_row < global_grid.blocks_per_row &&
//                 neighbor_col >= 0 && neighbor_col < global_grid.blocks_per_row) {
//                 int neighbor_id = neighbor_row * global_grid.blocks_per_row + neighbor_col;

//                 if (block_ids.find(neighbor_id) != block_ids.end()) {
//                     tasks[block_ids[neighbor_id]].push_back(&blocks[block_ids[block_id]]);
//                 }
//             }
//         }
//     }
// }

// Initialize the simulation
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

    // size of block and # blocks per row in the GLOBAL grid
    global_grid.block_size = block_size * cutoff;
    global_grid.blocks_per_row = int(ceil(size / global_grid.block_size));
    total_blocks = global_grid.blocks_per_row * global_grid.blocks_per_row;
    int blocks_per_proc = total_blocks / num_procs;

    // Calculate the start and end block for this process
    start_block = rank * blocks_per_proc;
    end_block = (rank + 1) * blocks_per_proc;

    // Adjust for the last process to take up any remaining blocks
    if (rank == num_procs - 1) {
        end_block = total_blocks;
    }

    // resize the local_blocks vector to hold the blocks for this process
    printf("Creating blocks\n\n\n");
    create_blocks(start_block, end_block);

    // Iterate over all particles and assing to local blocks
    part_to_block.resize(num_parts);
    for (int i = 0; i < num_parts; ++i) {
        int block_id = global_grid.get_block_index(parts[i].x, parts[i].y);
        part_to_block[i] = block_id;
        
        if(block_ids.find(block_id) != block_ids.end()) {
            blocks[block_ids[block_id]].add_particle(i);
            parts[i].ax = parts[i].ay = 0;
            // parts[i].id = block_id;
        }
    }
}


// iterate over the particles
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs){

    double* local_x_sums = new double[num_parts];
    double* local_y_sums = new double[num_parts];

    double* global_x_sums = new double[num_parts];
    double* global_y_sums = new double[num_parts];

    printf("Simulating one step\n\n\n");
    for (int i = 0; i < num_parts; ++i) {
        int block_id = part_to_block[i];

        
        if (block_ids.find(block_id) == block_ids.end()) {
            local_x_sums[i] = 0.0;
            local_y_sums[i] = 0.0;
            continue;
        }

        // compute forces of this particle with particles in the same block or neighboring blocks
        for (block* neighbor_block : tasks[block_ids[block_id]]) {
            // apply forces between particles in the current block and the neighbor block
            for (int j : neighbor_block->particles) {
                apply_force(parts[i], parts[j]);
            }
        }
        local_x_sums[i] = parts[i].ax;
        local_y_sums[i] = parts[i].ay;
    }


    // Perform the all-reduce operation to sum up x and y positions across all processes
    MPI_Allreduce(local_x_sums, global_x_sums, num_parts, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_y_sums, global_y_sums, num_parts, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        parts[i].ax = global_x_sums[i];
        parts[i].ay = global_y_sums[i];

        // move particle
        move(parts[i], size);
        parts[i].ax = parts[i].ay = 0;
    }

    update_blocks(parts, num_parts);

    delete[] local_x_sums;
    delete[] local_y_sums;
    delete[] global_x_sums;
    delete[] global_y_sums;
}


void apply_force_blocks(block* block1, block* block2, particle_t* parts) {
    for (int i : block1->particles) {
        for (int j : block2->particles) {
            if (i==j) continue;
            apply_force(parts[i], parts[j]);
        }
    }
}


void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // TODO
}
