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
 *      2. MPI_Send() : https://hpc-tutorials.llnl.gov/mpi/MPI_appendix/MPI_Send.txt
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
#include "functions.h"
#include <cuda_runtime_api.h>
#include <iostream>
using namespace std;



int main(int argc, char * argv[])
{
    int i = 0;                  // Indice
    int taskID = -1;
    int ntasks = -1;
    int sendcode = 0;
    int recvcode = 0;
    int msgtag = 42;           // Message tag used, see : https://stackoverflow.com/a/31471570/4021436
    int destidx = -1;
    int recvidx = -1;
    char hostname[250];
    //MPI_Status status; 
    hostname[249]='\0';

    /* MPI Initializations */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskID);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    // https://hpc-tutorials.llnl.gov/mpi/non_blocking/
    MPI_Request reqs[2];   // required variable for non-blocking calls
    //MPI_Status stats[2];   // required variable for Waitall routine
    gethostname(hostname, 1023);
    printf("Hello World from Task %i on %s\n", taskID,hostname);

    // Allocate 1GB of arrays
    //int N       = 15000 * 15000;        // 15000**2 * 4 bytes / int ~ 1GB
    // Only work with square matrices to make my life easier
    int nx      = 10;
    int ny      = 10;
    int N       = nx * ny; 
    float * sendA = (float *)malloc(sizeof(float) * nx * ny); // send to task + 1
    float * recvA = (float *)malloc(sizeof(float) * nx * ny); // receive from task - 1
    float * resA = (float *)malloc(sizeof(float) * nx * ny);  // result of mult sendA and recvA
    int dim[] = {nx, ny};
    
    initialize_matrix(recvA, dim, 0.0);
    identity_matrix(sendA, dim, (float)(taskID+1));


    printf("Starting Transfer...\n");
    // Transfer 1GB - 1000 times
    for(i=0; i<1000; i++){
        // Print progress
        if(taskID == 0 && i%10 == 0){
            printf("\ti = %i\n", i);
            fflush(stdout);
        }
        if(i%100 == 0){
            printf("\tTask %i : destidx = %i, recvidx = %i\n", taskID, destidx, recvidx);
            fflush(stdout);
        }

        // Round Robin strategy 0 -> 1 -> 2 -> ... -> N -> 0
        // destination...
        if(taskID == ntasks - 1){
            destidx = 0;
        }else{
            destidx = taskID + 1;
        }
        // recieve
        if(taskID == 0){
            recvidx = ntasks - 1;
        }else{
            recvidx = taskID - 1;
        }

        sendcode = MPI_Isend(sendA, N, MPI_FLOAT, destidx, msgtag, MPI_COMM_WORLD, &reqs[0]);
        recvcode = MPI_Irecv(recvA, N, MPI_FLOAT, recvidx, msgtag, MPI_COMM_WORLD, &reqs[1]);
        /* if(taskID == 0){
            errCode = MPI_Send(src0A, N, MPI_INT, 1, msgtag, MPI_COMM_WORLD);
            if(errCode != MPI_SUCCESS){
                fprintf(stderr, "ERROR!!! MPI_SUCCESS not achieved. See %i", errCode);
            }
        }
        if(taskID == 1){
            errCode = MPI_Recv(des1A, N, MPI_INT, 0, msgtag, MPI_COMM_WORLD, &status);
            if(errCode != MPI_SUCCESS){
                fprintf(stderr, "ERROR!!! MPI_SUCCESS not achieved. See %i", errCode);
            }
        }
        */
        MPI_Barrier(MPI_COMM_WORLD);    // Ensure every task completes
        if(sendcode != MPI_SUCCESS || recvcode != MPI_SUCCESS){
            fprintf(stderr, "ERROR!!! MPI_SUCCESS not achieved. See sendcode %i or recvcode %i", sendcode, recvcode);
        }
        // Visual Test that my code is assigning arrays and sending them correctly
        if(taskID == 1 && i%100 == 0){
            printf("task 1 : send array\n");
            print_1D_array(sendA, nx, ny);
            printf("task 1 : recv array\n");
            print_1D_array(recvA, nx, ny);
        }
    }

    MPI_Finalize();
    if(taskID == 0){
        printf("The transfer is finished!\n");
    }
    return 0;
}






