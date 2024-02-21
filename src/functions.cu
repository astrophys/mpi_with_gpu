/**************************************************************
 *  AUTHOR : Ali Snedden
 *  DATE   : 21-feb-2024
 *  LICENSE: GPL-3
 *  PURPOSE: To refresh myself on how using MPI. Let's do a 
 *           numerical integration using the trapezoidal rule.
 *  NOTES  :
 *      1. To compile and run : 
 *          $ ml load MVAPICH2/2.3.4
 *          $ mpicc mpi_integrate.c
 *          $ srun --ntasks=2 --nodes=2 --pty bash
 *          $ mpirun -np 2 ./a.out
 *
 *      2. To debug :
 *          $ ml load MVAPICH2/2.3.4
 *          $ mpicc -ggdb mpi_integrate.c
 *          $ mpirun -np 2 xterm -e gdb ./a.out
 *
 *  REFERNCES :
 *      1. MPI_Recv() : https://computing.llnl.gov/tutorials/mpi/man/MPI_Recv.txt
 *      2. MPI_Send() : https://computing.llnl.gov/tutorials/mpi/man/MPI_Send.txt
 *      3. gethostname() : https://stackoverflow.com/a/22207175/4021436
 *      4. MPI tags   : https://stackoverflow.com/a/31471570/4021436
 *      4. Debugging  : https://stackoverflow.com/a/2364825/4021436
 *
 *
 *
 *************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <cuda_runtime_api.h>
#include <iostream>
using namespace std;


// This is C++ code - from stackoverflow : https://stackoverflow.com/q/14038589 
/********************************************************
    ARGS:
        cudaError_t code
        const char* file : 
        int line :
    DESCRIPTION:
        Uses macro and inline function b/c it is important to preserve the
        file and line number in the error printing.
    RETURN:
    DEBUG:
    NOTES: 
    FUTURE:
*******************************************************/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s : %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}




