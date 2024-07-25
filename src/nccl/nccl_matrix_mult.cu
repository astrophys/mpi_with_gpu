/**************************************************************
 *  AUTHOR : Ali Snedden
 *  DATE   : 07/15/24
 *  LICENSE: GPL-3
 *  PURPOSE: To use my MPI code and adapting it to use NVidia's NCCL library
 *           This code does numerical integration using the trapezoidal rule.
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
 *      1. NCCL Example : https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
 *      2.
 *      3. 
 *      4.
 *      5. 
 *
 *
 *
 *************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <nccl.h>
//#include <unistd.h>
//#include <math.h>
//#include <time.h>
//#include "functions.h"
//#include "cpu_mult.h"
//#include "gpu_mult.h"
//#include <iostream>
using namespace std;

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)



int main(int argc, char * argv[])
{
    /***********************************************************/
    /****************** Variable Declaration *******************/
    /***********************************************************/
    bool verbose;
    char hostname[250];


    ncclComm_t comms[4];
    //managing 4 devices
    int nDev = 4;
    int size = 32*1024*1024;
    int devs[4] = { 0, 1, 2, 3 };

    //allocating and initializing device buffers
    float** sendbuff = (float**)malloc(nDev * sizeof(float*));
    float** recvbuff = (float**)malloc(nDev * sizeof(float*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);


    for (int i = 0; i < nDev; ++i) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaMalloc((void**)sendbuff + i, size * sizeof(float)));
      CUDACHECK(cudaMalloc((void**)recvbuff + i, size * sizeof(float)));
      CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
      CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
      CUDACHECK(cudaStreamCreate(s+i));
    }


    //initializing NCCL
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

    //calling NCCL communication API. Group API is required when using
    //multiple devices per thread
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i)
      NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
          comms[i], s[i]));
    NCCLCHECK(ncclGroupEnd());

    //synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaStreamSynchronize(s[i]));
    }

    //free device buffers
    for (int i = 0; i < nDev; ++i) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaFree(sendbuff[i]));
      CUDACHECK(cudaFree(recvbuff[i]));
    }

    //finalizing NCCL
    for(int i = 0; i < nDev; ++i)
        ncclCommDestroy(comms[i]);

    printf("Success \n");
    return 0;
}






