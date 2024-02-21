# mpi_with_gpu
Simple code to test integrating CUDA / OpenAcc with MPI

### Compile 

```
ml load openmpi/gcc/64/4.1.2
ml load cuda11.8/toolkit/11.8.0
make
salloc --cpus-per-task=2 --nodes=2
mpiexec -np 2 mpi_integrate
```
