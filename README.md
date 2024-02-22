# mpi_with_gpu
Simple code to test various combinations of multi-gpu computation. 
The goal is to observe the impact that the NVLink backplane has on the DGX servers.
I plan on doing that by running MPI + CUDA spread across the same machine
(and multiple machines) vs. taking advantage of the NVLink backplane that the DGX
server has.


### Compile and Run

Testing mpi + cpu computation
```
ml load openmpi/gcc/64/4.1.2
ml load cuda11.8/toolkit/11.8.0
make
salloc --cpus-per-task=2 --nodes=2
time mpiexec -np 2 mpi_matrix_mult
```

### 


### Refs
1. [Multi-GPU programming for earth scientists](https://www2.cisl.ucar.edu/sites/default/files/2022-07/Multi%20Node%20Multi%20GPU%20Programming.pdf)

2. [LLNL's amazing MPI Documentation](https://hpc-tutorials.llnl.gov/mpi)
