#include "common.h"
#include <cmath>

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

// Global grid instance
grid simulation_grid;
void init_simulation(particle_t* parts, int num_parts, double size) {
    simulation_grid.block_size = 12 * cutoff;
    simulation_grid.blocks_per_row = int(ceil(size / simulation_grid.block_size));

    // initialize (blocks_per_row x blocks_per_row) grid of blocks representing the xy plane
    simulation_grid.blocks.resize(simulation_grid.blocks_per_row * simulation_grid.blocks_per_row);

    // Assign particles to respective blocks based on `x` and `y` coordinates
    for (int i = 0; i < num_parts; ++i) {
        int block_id = simulation_grid.get_block_index(parts[i].x, parts[i].y);
        // simulation_grid.blocks[block_id].insert(i);
        simulation_grid.blocks[block_id].add_particle(i);
        parts[i].block_id = block_id;
        parts[i].ax = parts[i].ay = 0;
    }        
}

// iterate over the particles
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // for each particle, 
    for (int i = 0; i < num_parts; ++i) {

        // get the block where the particle is located
        int block_id = simulation_grid.get_block_index(parts[i].x, parts[i].y);

        // compute forces of this particle with particles in the same block or neighboring blocks
                //   TODO: reduce the comparisons to neighbouring blocks and/or change to recursive implementation (block within block)
        for (int drow = -1; drow <= 1; drow++) {
            for (int dcol = -1; dcol <= 1; dcol++) {
                
                int neighbor_block_row = block_id / simulation_grid.blocks_per_row + drow;
                int neighbor_block_col = block_id % simulation_grid.blocks_per_row + dcol;

                // if neighbor block within bounds, apply forces
                if (neighbor_block_row >= 0 && neighbor_block_row < simulation_grid.blocks_per_row &&
                    neighbor_block_col >= 0 && neighbor_block_col < simulation_grid.blocks_per_row) {
                    
                    // get the neighbor block id
                    int neighbor_block_id = neighbor_block_row * simulation_grid.blocks_per_row + neighbor_block_col;

                    // apply forces between particles in the current block and the neighbor block
                    for (int j : simulation_grid.blocks[neighbor_block_id].particles) {
                        apply_force(parts[i], parts[j]);
                    }
                }
            }
        }
    }

    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        // move particle
        move(parts[i], size);
        parts[i].ax = parts[i].ay = 0;
    }
    simulation_grid.update_blocks(parts, num_parts);
}



// iterate over the blocks
// void simulate_one_step(particle_t* parts, int num_parts, double size) {
//     // Iterate over each block
//     for (int block_id = 0; block_id < simulation_grid.blocks.size(); ++block_id) {
//         // Compute forces within the current block and with neighboring blocks
//         for (int drow = -1; drow <= 1; drow++) {
//             for (int dcol = -1; dcol <= 1; dcol++) {
//                 int neighbor_block_row = (block_id / simulation_grid.blocks_per_row) + drow;
//                 int neighbor_block_col = (block_id % simulation_grid.blocks_per_row) + dcol;

//                 // Check if the neighbor block is within bounds
//                 if (neighbor_block_row >= 0 && neighbor_block_row < simulation_grid.blocks_per_row &&
//                     neighbor_block_col >= 0 && neighbor_block_col < simulation_grid.blocks_per_row) {
                    
//                     int neighbor_block_id = neighbor_block_row * simulation_grid.blocks_per_row + neighbor_block_col;

//                     // Apply forces between particles in the current block and the neighbor block
//                     for (int i : simulation_grid.blocks[block_id].particles) {
//                         for (int j : simulation_grid.blocks[neighbor_block_id].particles) {
//                             apply_force(parts[i], parts[j]);
//                             }
//                         }
//                     }
//                 }
//             }
//         }

//     // Move Particles
//     for (int i = 0; i < num_parts; ++i) {
//         // move particle
//         move(parts[i], size);
//         parts[i].ax = parts[i].ay = 0;
//         simulation_grid.update_particle_block(i, parts);
//     }
    
    // simulation_grid.update_blocks(parts, num_parts);
// }
