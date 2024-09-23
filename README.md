# mpi_with_gpu
Simple code to test various combinations of multi-gpu computation. 
The goal is to observe the impact that the NVLink backplane has on the DGX servers.
I plan on doing that by running MPI + CUDA spread across the same machine
(and multiple machines) vs. taking advantage of the NVLink backplane that the DGX
server has.


### Compile and Run

Testing mpi + cpu computation
```
ml load openmpi/gcc/64/4.1.5
ml load cuda11.8/toolkit/11.8.0
make
salloc --ntasks=2 --cpus-per-task=2 --nodes=2 
time mpiexec -np 2 mpi_matrix_mult --option mpi_cpu --size 15 --verbose true
```

Testing mpi + gpu computation
```
ml load openmpi/gcc/64/4.1.5
ml load cuda11.8/toolkit/11.8.0
make
salloc --ntasks=2 --partition=dgxq --cpus-per-task=2 --gres=gpu --nodes=2
time mpiexec -np 2 mpi_matrix_mult --option mpi_cpu --size 15 --verbose true
```

If running on DGXs, you'll want to use the [Nvidia HPC-SDK](https://developer.nvidia.com/hpc-sdk).  It provides module files and a comprensive set of necessary software.
### 


### Refs
1. [Multi-GPU programming for earth scientists](https://www2.cisl.ucar.edu/sites/default/files/2022-07/Multi%20Node%20Multi%20GPU%20Programming.pdf)

2. [LLNL's amazing MPI Documentation](https://hpc-tutorials.llnl.gov/mpi)

3. [SLURM MPI Guide](https://slurm.schedmd.com/mpi_guide.html)
