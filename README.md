# QUBO-Benchmarking

A comprehensive benchmarking framework for solving Quadratic Unconstrained Binary Optimization (QUBO) problems running on the GH200, currently for Max-Cut problems and G-Set benchmark instances.

## Overview

This repository provides tools for:
- **GPU-accelerated Simulated Annealing** using CuPy for parallel optimization of simulated annealing
- **D-Wave Quantum Annealing** via Ocean SDK (hybrid and quantum samplers)
- **Classical solver** Using D-Wave-neal (simulated annealing)
- **Benchmarking** currently on G-Set Max-Cut instances

### Benchmark
- **G-Set Instances**: Automatic download and parsing of standard Max-Cut benchmarks
- **Solution Quality**: Approximation ratio analysis against best known solution
- **Scalability Testing**: Performance across the full problem size (800 to 20,000+ nodes)
- **Comparative Analysis**: Side-by-side solver performance evaluation

## Quick Start

### Prerequisites
```bash
# GPU acceleration (optional but recommended)
pip install cupy-cuda11x  # or cupy-cuda12x for CUDA 12

# D-Wave Ocean SDK
pip install dwave-ocean-sdk

# Standard dependencies
pip install numpy networkx matplotlib pandas requests
```
### Basic Usage

#### GPU Simulated Annealing
```python
# Load G-Set instance and solve with GPU SA
adj_matrix = load_gset_instance('G14')  # 800 nodes, 4694 edges
maxcut_qubo = MaxCutQUBO(adj_matrix)
Q_gpu = cp.asarray(maxcut_qubo.Q)

gpu_sa = GPUSimulatedAnnealing(Q_gpu, n_runs=800)
best_solution, best_energy = gpu_sa.solve(max_iter=6000)
```

#### D-Wave Quantum Solver
```python
# Set up D-Wave solver
solver = MaxCutQUBOSolver(api_token="your-dwave-token")
solver.load_gset_graph('G22')  # 2000 nodes
solver.formulate_qubo()
results = solver.solve(time_limit=20)
```

#### Comprehensive Benchmarking
```python
# Run the benchmark across multiple instances
benchmark = MaxCutBenchmark()
results = benchmark.run_systematic_benchmark(
    instance_list=[11, 12, 13, 14, 43], 
    num_runs=5
)
```

## Example G-Set Benchmark Results

| Instance | Nodes | Method | Cut Value | Best Known | Quality |
|----------|-------|--------|-----------|------------|---------|
| G14      | 800   | Hybrid | 3064      | 3064       | 100%    |
| G22      | 2000  | D-Wave | 13359     | 13359      | 100%    |
| G43      | 1000  | GPU SA | 6650      | 6660       | 99.8%   |


This framework is designed for extensibility:
- Add new QUBO formulations beyond Max-Cut
- Implement additional quantum/classical solvers
- Extend benchmarking to other quantum hardware simulators
- Contribute performance optimizations
