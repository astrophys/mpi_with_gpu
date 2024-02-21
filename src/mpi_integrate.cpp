/**************************************************************
 *  AUTHOR : Ali Snedden
 *  DATE   : 12/29/20
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



int main(int argc, char * argv[])
{
    int i = 0;                  // Indice
    int taskID = -1;
    int ntasks = -1;
    int errCode = 0;
    int messTag = 42;           // Message tag used, see : https://stackoverflow.com/a/31471570/4021436
    char hostname[250];
    MPI_Status status; 
    hostname[249]='\0';

    /* MPI Initializations */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskID);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    gethostname(hostname, 1023);
    printf("Hello World from Task %i on %s\n", taskID,hostname);

    // Allocate 1GB of arrays
    int N       = 250e6;        // 250e6 * 4 bytes / int = 1GB
    int * src0A = (int *)malloc(sizeof(int) * N);   // Task 0 : source array
    int * des1A = (int *)malloc(sizeof(int) * N);   // Task 1 : destination array
    
    for(i=0; i<N; i++){
        src0A[i] = 10;
        des1A[i] = 0; 
    }

    printf("Starting Transfer...\n");
    // Transfer 1GB - 1000 times
    for(i=0; i<1000; i++){
        // Print progress
        if(taskID == 0 && i%10 == 0){
            printf("\ti = %i\n", i);
            fflush(stdout);
        }
        if(taskID == 0){
            errCode = MPI_Send(src0A, N, MPI_INT, 1, messTag, MPI_COMM_WORLD);
            if(errCode != MPI_SUCCESS){
                fprintf(stderr, "ERROR!!! MPI_SUCCESS not achieved. See %i", errCode);
            }
        }
        if(taskID == 1){
            errCode = MPI_Recv(des1A, N, MPI_INT, 0, messTag, MPI_COMM_WORLD, &status);
            if(errCode != MPI_SUCCESS){
                fprintf(stderr, "ERROR!!! MPI_SUCCESS not achieved. See %i", errCode);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);    // Ensure every task completes
    }

    MPI_Finalize();
    if(taskID == 0){
        printf("The transfer is finished!\n");
    }
    return 0;
}






