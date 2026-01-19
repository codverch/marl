// STD Headers
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <atomic>
#include <fstream>
#include <iomanip>

// MARL Headers
#include "marl/defer.h"
#include "marl/event.h"
#include "marl/scheduler.h"
#include "marl/waitgroup.h"

// ============================================================================
// MicrobenchmarkMetrics: Tracks Performance Metrics
// ============================================================================
/**
 * Aggregates performance metrics across all microbenchmark tasks.
 * Uses atomic operations for thread-safe updates from multiple fibers.
 */
struct MicrobenchmarkMetrics {
    // Thread-safe counters for various metrics
    std::atomic<uint64_t> total_fiber_switches{0};       // Count of fiber switches (user-space)
    std::atomic<uint64_t> total_tasks_completed{0};      // Total tasks finished
    std::atomic<uint64_t> total_computation_time_ns{0};  // Aggregate computation time
    std::atomic<uint64_t> total_blocking_time_ns{0};     // Aggregate blocking time
    
    // Timing for overall microbenchmark duration
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    
    // Enable move semantics for struct with atomic members
    MicrobenchmarkMetrics() = default;
    MicrobenchmarkMetrics(MicrobenchmarkMetrics&& other) noexcept
        : total_fiber_switches(other.total_fiber_switches.load()),
          total_tasks_completed(other.total_tasks_completed.load()),
          total_computation_time_ns(other.total_computation_time_ns.load()),
          total_blocking_time_ns(other.total_blocking_time_ns.load()),
          start_time(other.start_time),
          end_time(other.end_time) {}
    
    MicrobenchmarkMetrics& operator=(MicrobenchmarkMetrics&& other) noexcept {
        total_fiber_switches.store(other.total_fiber_switches.load());
        total_tasks_completed.store(other.total_tasks_completed.load());
        total_computation_time_ns.store(other.total_computation_time_ns.load());
        total_blocking_time_ns.store(other.total_blocking_time_ns.load());
        start_time = other.start_time;
        end_time = other.end_time;
        return *this;
    }
    
    // Delete copy operations since atomics can't be copied
    MicrobenchmarkMetrics(const MicrobenchmarkMetrics&) = delete;
    MicrobenchmarkMetrics& operator=(const MicrobenchmarkMetrics&) = delete;
    
    /**
     * Calculates throughput in tasks per second.
     * 
     * @return Throughput (tasks/sec), or 0.0 if duration is invalid.
     */
    double get_throughput() const {
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        return (duration_ms > 0) 
            ? (total_tasks_completed.load() * 1000.0 / duration_ms) 
            : 0.0;
    }
    
    /**
     * Calculates average computation time per task.
     * 
     * @return Average computation time in milliseconds.
     */
    double get_avg_computation_time_ms() const {
        uint64_t count = total_tasks_completed.load();
        
        return (count > 0) 
            ? (total_computation_time_ns.load() / (count * 1000000.0)) 
            : 0.0;
    }
};

// ============================================================================
// Computation Simulation: Cache and Branch Predictor Pollution
// ============================================================================
/**
 * Simulates computational work that pollutes CPU microarchitectural structures.
 * 
 * This function intentionally creates cache pollution and branch mispredictions
 * to simulate real data center applications. The pollution effects are what make
 * fiber switches expensive.
 * 
 * Cache Pollution:
 * - Uses a 64KB array (typical L1 cache size)
 * - Random memory accesses destroy spatial locality
 * - Each fiber has its own array, so fiber switches evict cache lines
 * 
 * Branch Predictor Pollution:
 * - Unpredictable modulo-based branches
 * - Different fibers have different branch patterns
 * - Fiber switches confuse the branch predictor
 * 
 * @param complexity Number of iterations (controls computation duration)
 * @param computation_time_ns Output parameter for measured execution time
 */
void do_computation(int complexity, uint64_t& computation_time_ns) {
    auto start = std::chrono::steady_clock::now();
    
    // Array size matches typical L1 data cache (32-64KB)
    // This ensures we fill the cache and cause evictions
    const int ARRAY_SIZE = 1024 * 64;  // int = 4 bytes, 64K ints = 256KB
    static thread_local std::vector<int> data(ARRAY_SIZE);
    
    // Thread-local random number generator for unpredictable access patterns
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dis(0, ARRAY_SIZE - 1);
    
    // Volatile prevents compiler from optimizing away the computation
    volatile int result = 0;
    
    // Main computation loop with cache and branch predictor pollution
    for (int i = 0; i < complexity; ++i) {
        // Random index access (destroys spatial locality)
        int idx = dis(gen);
        
        // Unpredictable branch patterns (confuses branch predictor)
        // Using modulo creates three branches with equal probability
        if (data[idx] % 3 == 0) {
            data[idx] = data[idx] * 2 + 1;
        } 
        else if (data[idx] % 3 == 1) {
            data[idx] = data[idx] / 2 + 3;
        } 
        else {
            data[idx] = data[idx] ^ 0x5A5A;  // XOR with pattern
        }
        
        // Accumulate result to prevent dead code elimination
        result += data[idx];
        
        // Additional random access for more cache pressure
        int next_idx = (idx + data[idx]) % ARRAY_SIZE;
        result ^= data[next_idx];
    }
    
    auto end = std::chrono::steady_clock::now();
    computation_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - start).count();
}

// ============================================================================
// I/O Simulation: Blocking Operation That Triggers Fiber Switch
// ============================================================================
/**
 * Simulates an I/O operation that blocks the current fiber.
 * 
 * Models modern data center I/O devices with microsecond-scale latency:
 * - NVMe SSDs: ~10-100 microseconds
 * - GPU/Accelerator operations: ~10-500 microseconds
 * - Local network RPC: ~50-200 microseconds
 * 
 * When a fiber blocks waiting for I/O, the MARL scheduler switches to another
 * ready fiber in USER SPACE (no kernel involvement). This fiber switch causes 
 * microarchitectural pollution:
 * - The blocked fiber's data is evicted from cache
 * - The new fiber's branch patterns train the predictor differently
 * - When the blocked fiber resumes, it experiences a "cold start"
 * 
 * This is the key mechanism that causes oversubscription overhead.
 * 
 * @param blocking_duration_us Duration to block in microseconds (simulates I/O latency)
 * @param blocking_time_ns Output parameter for measured blocking time
 */
void do_blocking_operation(int blocking_duration_us, uint64_t& blocking_time_ns) {
    auto start = std::chrono::steady_clock::now();
    
    // MARL event for fiber synchronization
    auto event = std::make_shared<marl::Event>();
    
    // Spawn background thread to simulate I/O completion
    // The detached thread signals the event after a delay
    std::thread([event, blocking_duration_us]() {
        std::this_thread::sleep_for(std::chrono::microseconds(blocking_duration_us));
        event->signal();  // Signal completion
    }).detach();
    
    // Block current fiber until event is signaled
    // During this wait, MARL scheduler switches to another fiber (USER-SPACE SWITCH)
    event->wait();
    
    auto end = std::chrono::steady_clock::now();
    blocking_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - start).count();
}

// ============================================================================
// Task Execution: Combines Computation and I/O
// ============================================================================
/**
 * Represents a single task in the microbenchmark.
 * 
 * Each task performs multiple iterations of:
 * 1. Computation (with cache/branch predictor pollution)
 * 2. Occasional I/O blocking (configurable probability)
 * 
 * This mix simulates real data center applications that combine CPU-intensive
 * operations with I/O waits (NVMe, accelerators, local network).
 * 
 * Fiber Switch Detection:
 * We track switches using thread-local storage. When execution switches
 * from one fiber to another (in user-space), we detect and count it.
 * 
 * @param task_id Unique identifier for this task
 * @param num_iterations Number of computation/I/O cycles to perform
 * @param io_probability Probability of I/O operation (0.0 to 1.0)
 * @param metrics Shared metrics structure for recording results
 */
void run_task(int task_id, int num_iterations, double io_probability, 
              MicrobenchmarkMetrics& metrics) {
    // Thread-local variable to detect fiber switches
    // Each OS thread maintains its own copy of this variable
    static thread_local int last_fiber_id = -1;
    
    int current_fiber_id = task_id;
    
    // Detect fiber switch: if we're on the same OS thread but different fiber
    if (last_fiber_id != -1 && last_fiber_id != current_fiber_id) {
        // Fiber switch detected! Increment counter.
        // This happens in user-space (no kernel involvement)
        metrics.total_fiber_switches.fetch_add(1, std::memory_order_relaxed);
    }
    last_fiber_id = current_fiber_id;
    
    // Per-task random number generator (seeded with task_id for reproducibility)
    std::mt19937 gen(task_id);
    std::uniform_int_distribution<> comp_dis(1000, 5000);   // Computation complexity
    std::uniform_int_distribution<> block_dis(10, 100);     // Blocking duration (microseconds)
    std::uniform_real_distribution<> prob_dis(0.0, 1.0);    // Probability distribution
    
    // Execute multiple iterations of work
    for (int i = 0; i < num_iterations; ++i) {
        uint64_t comp_time_ns = 0;
        uint64_t block_time_ns = 0;
        
        // Phase 1: Computation (always happens)
        int complexity = comp_dis(gen);
        do_computation(complexity, comp_time_ns);
        metrics.total_computation_time_ns.fetch_add(comp_time_ns, std::memory_order_relaxed);
        
        // Phase 2: I/O blocking (configurable probability)
        // This simulates NVMe reads, accelerator calls, local network RPC, etc.
        if (prob_dis(gen) < io_probability) {
            int block_duration = block_dis(gen);
            do_blocking_operation(block_duration, block_time_ns);
            metrics.total_blocking_time_ns.fetch_add(block_time_ns, std::memory_order_relaxed);
            
            // After blocking, there was definitely a fiber switch
            metrics.total_fiber_switches.fetch_add(1, std::memory_order_relaxed);
        }
        
        // Record task completion
        metrics.total_tasks_completed.fetch_add(1, std::memory_order_relaxed);
    }
}

// ============================================================================
// Microbenchmark Execution: Orchestrates Tasks with MARL
// ============================================================================
/**
 * Runs the microbenchmark with a specified number of fibers.
 * 
 * This function:
 * 1. Configures MARL scheduler with fixed worker threads
 * 2. Spawns the requested number of tasks (fibers)
 * 3. Waits for all tasks to complete
 * 4. Returns aggregated metrics
 * 
 * The key insight: We keep worker threads (OS threads) constant while varying
 * the number of fibers. This creates oversubscription when num_fibers >> num_workers.
 * 
 * @param num_fibers Number of fibers to spawn
 * @param iterations_per_fiber Work per fiber
 * @param io_probability Probability of I/O operation (0.0 to 1.0)
 * @param num_worker_threads Number of OS worker threads (should be fixed)
 * @return Aggregated metrics for this microbenchmark run
 */
MicrobenchmarkMetrics run_microbenchmark(int num_fibers, 
                                          int iterations_per_fiber,
                                          double io_probability,
                                          int num_worker_threads) {
    MicrobenchmarkMetrics metrics;
    
    // Configure MARL scheduler
    marl::Scheduler::Config config;
    config.setWorkerThreadCount(num_worker_threads);
    
    // Create scheduler instance
    auto scheduler = std::make_unique<marl::Scheduler>(config);
    scheduler->bind();   // Bind scheduler to current thread
    defer(scheduler->unbind());  // Ensure unbind happens at scope exit
    
    // Record microbenchmark start time
    metrics.start_time = std::chrono::steady_clock::now();
    
    // WaitGroup for synchronization
    // Counter starts at num_fibers, decrements to 0
    marl::WaitGroup wg(num_fibers);
    
    // Spawn all tasks as MARL fibers
    for (int i = 0; i < num_fibers; ++i) {
        marl::schedule([i, iterations_per_fiber, io_probability, &metrics, wg] {
            // Ensure wg.done() is called even if task throws
            defer(wg.done());
            
            // Execute task
            run_task(i, iterations_per_fiber, io_probability, metrics);
        });
    }
    
    // Block until all tasks complete
    wg.wait();
    
    // Record microbenchmark end time
    metrics.end_time = std::chrono::steady_clock::now();
    
    return metrics;
}

// ============================================================================
// Result Storage: Structured Result for CSV Output
// ============================================================================
/**
 * Stores results for a single microbenchmark configuration.
 * Used for CSV output and analysis.
 */
struct MicrobenchmarkResult {
    int num_fibers;                     // Number of fibers spawned
    int num_workers;                    // Number of OS worker threads
    double throughput;                  // Tasks per second
    uint64_t fiber_switches;            // Count of fiber switches (user-space)
    double avg_computation_time_ms;     // Average computation time
    double total_time_ms;               // Total microbenchmark duration
    double oversubscription_ratio;      // num_fibers / num_workers
    double io_probability;              // I/O operation probability
};

// ============================================================================
// CSV Export: Write Results for Visualization
// ============================================================================
/**
 * Writes microbenchmark results to CSV file for analysis and visualization.
 * 
 * Output format:
 * num_threads,num_workers,oversubscription_ratio,throughput,thread_switches,
 * avg_computation_time_ms,total_time_ms,io_probability
 * 
 * Note: Column names use "num_threads" and "thread_switches" for compatibility
 * with existing visualization scripts, but these represent fibers and fiber switches.
 * 
 * @param results Vector of microbenchmark results
 * @param filename Output CSV filename
 */
void write_results_csv(const std::vector<MicrobenchmarkResult>& results, 
                       const std::string& filename) {
    std::ofstream file(filename);
    
    // Write CSV header (using legacy names for compatibility)
    file << "num_threads,num_workers,oversubscription_ratio,throughput,thread_switches,"
         << "avg_computation_time_ms,total_time_ms,io_probability\n";
    
    // Write data rows
    for (const auto& r : results) {
        file << r.num_fibers << ","
             << r.num_workers << ","
             << std::fixed << std::setprecision(2) << r.oversubscription_ratio << ","
             << std::fixed << std::setprecision(2) << r.throughput << ","
             << r.fiber_switches << ","
             << std::fixed << std::setprecision(4) << r.avg_computation_time_ms << ","
             << std::fixed << std::setprecision(2) << r.total_time_ms << ","
             << std::fixed << std::setprecision(2) << r.io_probability << "\n";
    }
    
    std::cout << "Results written to " << filename << std::endl;
}

// ============================================================================
// Main: Microbenchmark Driver
// ============================================================================
/**
 * Main microbenchmark driver.
 * 
 * Tests multiple fiber counts from undersubscribed (0.5x cores) to highly
 * oversubscribed (62.5x cores) to demonstrate performance degradation.
 * 
 * Expected Results:
 * - Performance peaks around hardware concurrency (1x)
 * - Degradation begins at 2-4x oversubscription
 * - Significant loss at 8x+ oversubscription
 * - Proves that oversubscription hurts performance (especially with microsecond I/O)
 */
int main(int argc, char** argv) {
    std::cout << "================================================\n";
    std::cout << "Fiber Oversubscription Microbenchmark (MARL)\n";
    std::cout << "NVMe/Accelerator I/O Profile (microseconds)\n";
    std::cout << "User-space fiber switches (no kernel overhead)\n";
    std::cout << "================================================\n\n";
    
    // ========================================================================
    // Configuration
    // ========================================================================
    const int ITERATIONS_PER_FIBER = 100;
    const int NUM_WORKER_THREADS = std::thread::hardware_concurrency();
    
    // I/O probability (configurable parameter)
    double io_probability = 0.3;  // 30% chance of I/O per iteration
    
    // Parse command line arguments
    if (argc > 1) {
        io_probability = std::stod(argv[1]);
        if (io_probability < 0.0 || io_probability > 1.0) {
            std::cerr << "Error: I/O probability must be between 0.0 and 1.0\n";
            return 1;
        }
    }
    
    std::cout << "Hardware concurrency: " << NUM_WORKER_THREADS << " cores\n";
    std::cout << "Iterations per fiber: " << ITERATIONS_PER_FIBER << "\n";
    std::cout << "I/O latency range: 10-100 microseconds (NVMe/Accelerator)\n";
    std::cout << "I/O probability: " << std::fixed << std::setprecision(1) 
              << (io_probability * 100) << "%\n\n";
    
    // ========================================================================
    // Fiber Count Scenarios
    // ========================================================================
    // Test various oversubscription levels to demonstrate performance impact
    std::vector<int> fiber_counts = {
        NUM_WORKER_THREADS / 2,       // 0.5x - Undersubscribed
        NUM_WORKER_THREADS,           // 1.0x - Balanced (optimal)
        NUM_WORKER_THREADS * 2,       // 2.0x - Moderate oversubscription
        NUM_WORKER_THREADS * 4,       // 4.0x - High oversubscription
        NUM_WORKER_THREADS * 8,       // 8.0x - Very high oversubscription
        NUM_WORKER_THREADS * 16,      // 16.0x - Extreme oversubscription
        NUM_WORKER_THREADS * 32,      // 32.0x - Datacenter-like scenario
        500,                          // Fixed large count
        1000                          // Fixed very large count
    };
    
    std::vector<MicrobenchmarkResult> results;
    
    // ========================================================================
    // Execute Microbenchmarks
    // ========================================================================
    for (int num_fibers : fiber_counts) {
        std::cout << "Running microbenchmark with " << num_fibers << " fibers...\n";
        
        // Run microbenchmark and collect metrics
        auto metrics = run_microbenchmark(num_fibers, ITERATIONS_PER_FIBER, 
                                          io_probability, NUM_WORKER_THREADS);
        
        // Package results
        MicrobenchmarkResult result;
        result.num_fibers = num_fibers;
        result.num_workers = NUM_WORKER_THREADS;
        result.throughput = metrics.get_throughput();
        result.fiber_switches = metrics.total_fiber_switches.load();
        result.avg_computation_time_ms = metrics.get_avg_computation_time_ms();
        result.total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            metrics.end_time - metrics.start_time).count();
        result.oversubscription_ratio = static_cast<double>(num_fibers) / NUM_WORKER_THREADS;
        result.io_probability = io_probability;
        
        results.push_back(result);
        
        // Print summary
        std::cout << "  Throughput: " << std::fixed << std::setprecision(2) 
                  << result.throughput << " tasks/sec\n";
        std::cout << "  Fiber switches: " << result.fiber_switches << "\n";
        std::cout << "  Oversubscription: " << std::fixed << std::setprecision(2) 
                  << result.oversubscription_ratio << "x\n";
        std::cout << "  Total time: " << result.total_time_ms << " ms\n\n";
    }
    
    // ========================================================================
    // Export Results
    // ========================================================================
    write_results_csv(results, "benchmark_results.csv");
    
    std::cout << "\n================================================\n";
    std::cout << "Microbenchmark complete!\n";
    std::cout << "Run visualization: python3 plot.py\n";
    std::cout << "================================================\n";
    
    return 0;
}