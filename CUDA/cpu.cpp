#include "common.h"
#include <cuda.h>
#define NUM_THREADS 256
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Static global variables
int blks;
int bs = 6;
double bin_size;
int num_bins_x;
int num_bins;
unsigned int memsize;

// device pointers
int* bin_counts_d;              // keeps track of the number of particles in each bin
int* bin_prefix_sum_d;          // keeps track of the start and end index of each bin
particle_t* sorted_particles_d; // particles sorted by bin ID

void count_particles_per_bin(particle_t* parts, int* bin_counts_d, int num_parts, double bin_size, int num_bins_x) {
    // each thread = particle participates in the sum, adding one/itself to the bin it belongs to
    
    // thread computes its unique id
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
      // e.g.: first thread in second block, 4 threads per block
      // blockIdx.x = 1, threadIdx.x = 0, blockDim.x = 4
      // idx = 1 * 4 + 0 = 4

    // guard statement because can have more threads than particles // (that is, nothing done by excess threads in the block added to process remainder if not divisible)
    if(tid >= num_parts) return;
    
    // compute bin indices for each particle
    int bin_x = parts[tid].x / bin_size;
    int bin_y = parts[tid].y / bin_size;
    int bin_idx = bin_x + bin_y * num_bins_x;
    
    // atomicAdd to avoid race conditions; more than one thread can be updating the same bin count
    atomicAdd(&bin_counts_d[bin_idx], 1);
}

__global__ void sort_particles_by_bin(particle_t* parts, particle_t* sorted_particles, int* bin_prefix_sum_d, int num_particles, double bin_size, int num_bins_x) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_particles) return;

    // find bin index for this particle
    int bin_x = parts[tid].x / bin_size;   // j 
    int bin_y = parts[tid].y / bin_size;   // i 
    int bin_idx =  bin_y * num_bins_x + bin_x;  // i * bins_per_row + j

    // find the index for this particle=thread within the bin and update the starting position
    int idx = atomicAdd(&bin_prefix_sum_d[bin_idx], 1);
      // e.g.: bin_prefix_sum = [0, 3, 4, 4, 6]  (that is, bin 1 have 3 particles = threads 0,1,2)
      //     t0 executes first; find its index as 0 and update starting position for bin0 to 1
      //     t2 executes; find its index as 1 and update starting position for bin0 to 2
      //     t1 executes; find its index as 2 and update starting position for bin0 to 3 

    // Place the particle in the sorted array
    sorted_particles[idx] = parts[tid];
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}


__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //  very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* sorted_particles, int* bin_prefix_sum, int num_particles, int num_bins_x, double bin_size) {
    
    // get particle
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_particles) return;

    particle_t& particle = sorted_particles[tid];
    particle.ax = particle.ay = 0;

    // bin indices of this particle
    int bin_x = int(particle.x / bin_size);
    int bin_y = int(particle.y / bin_size);
    // int bin_idx = bin_x + bin_y * num_bins_x;

    // Loop over neighboring bins
    for (int drow = -1; drow <= 1; drow++) {
        for (int dcol = -1; dcol <= 1; dcol++) {
            // int neighbor_bin_x = (bin_idx / num_bins_x) + drow;
            // int neighbor_bin_y = (bin_idx % num_bins_x) + dcol;
            int neighbor_bin_x = bin_x + dcol;
            int neighbor_bin_y = bin_y + drow;
        
            // if within bounds
            if (neighbor_bin_x >= 0 && neighbor_bin_x < num_bins_x &&
                neighbor_bin_y >= 0 && neighbor_bin_y < num_bins_x) {
                
                int neighbor_bin_idx = neighbor_bin_x + neighbor_bin_y * num_bins_x;
                int start_idx = bin_prefix_sum[neighbor_bin_idx];
                int end_idx = bin_prefix_sum[neighbor_bin_idx + 1];

                // Compute forces against all particles in the neighboring bin
                for (int idx = start_idx; idx < end_idx; idx++) {
                    apply_force_gpu(particle, sorted_particles[idx]);
                }
            }
        }
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    
    // set sufficient number of blocks to process all particles
    // obs: if num_parts not divisible by NUM_THREADS, some threads will be idle
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    // compute grid setting
    bin_size = bs * cutoff; // size of a bin of particles
    num_bins_x = ceil(size / bin_size); // number of bins per row
    num_bins = num_bins_x * num_bins_x; // total number of bins (square grid)
    memsize = (num_bins + 1) * sizeof(int);  // memory size for bin_counts and bin_prefix_sum

    // allocate memory on device for global variables
    cudaMalloc((void**)&bin_counts_d, memsize);
    cudaMalloc((void**)&bin_prefix_sum_d, memsize);
    cudaMalloc((void**)&sorted_particles_d, num_parts * sizeof(particle_t));
}

void prefix_sum(int* bin_counts_d, int* bin_prefix_sum_d, int num_bins) {
    // wraps device pointer with thrust pointers `dev_ptr_bin_counts`, `dev_ptr_bin_prefix_sum`
    thrust::device_ptr<int> d_ptr_bin_counts(bin_counts_d);  
    thrust::device_ptr<int> d_ptr_bin_prefix_sum(bin_prefix_sum_d);

    // compute prefix sum using thrust
    thrust::exclusive_scan(d_ptr_bin_counts, d_ptr_bin_counts + num_bins, d_ptr_bin_prefix_sum);
        // e.g.: bin_counts =     [3, 1, 0, 2, 4]
        //       bin_prefix_sum = [0, 3, 4, 4, 6] 
        // => 1st bin spans [0,3); 2nd bin [3,4); etc
}

// parts live in GPU memory
void simulate_one_step(particle_t* parts, int num_parts, double size) {

    // count particles per bin using GPU
    cudaMemset(bin_counts_d, 0, memsize); // set counts to zero
    count_particles_per_bin<<<blks, NUM_THREADS>>>(parts, bin_counts_d, num_parts, bin_size, num_bins_x);

    // compute prefix_sum using GPU
    prefix_sum(bin_counts_d, bin_prefix_sum_d, num_bins);
    // sort particles per bin
    sort_particles_by_bin<<<blks, NUM_THREADS>>>(parts, sorted_particles_d, bin_prefix_sum_d, num_parts, bin_size, num_bins_x);

    // recover prefix_sum for correct indexing in compute_forces 
    prefix_sum(bin_counts_d, bin_prefix_sum_d, num_bins);
    
    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(sorted_particles_d, bin_prefix_sum_d, num_parts, num_bins_x, bin_size);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);

}


// Clear allocations
void clear_simulation() {

}
