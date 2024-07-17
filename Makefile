# To compile :
#   ml load openmpi/gcc/64/4.1.2
#
#

# changed 7/9/2024
###### mpi_matrix_mult ######
MPI_LIB = -L/cm/local/apps/cuda/libs/current/lib64 -L/cm/shared/apps/openmpi4/gcc/4.1.5/lib -lmpi
CXXFLAGS = -Wall -g
PREFIX_MPI = src/mpi
CC = mpic++
NVCC = nvcc
NVCCFLAGS = --compiler-options ${CXXFLAGS} -I${CPATH} -I${CUDA_INC_PATH}/include

OBJ_DIR_MPI = $(PREFIX_MPI)/obj
OBJ_MPI = $(OBJ_DIR_MPI)/mpi_matrix_mult.o $(OBJ_DIR_MPI)/functions.o $(OBJ_DIR_MPI)/cpu_mult.o $(OBJ_DIR_MPI)/gpu_mult.o
#CU_OBJ = $(OBJ_DIR)/functions.o

# E.g. of working compilation nvcc -I/cm/shared/apps/openmpi4/gcc/4.1.2/include -I${CUDA_INC_PATH}/include -L/cm/local/apps/cuda/libs/current/lib64 -L/cm/shared/apps/openmpi4/gcc/4.1.2/lib -lmpi -o mpi_matrix_mult src/obj/mpi_matrix_mult.o


###### nccl_matrix_mult ######

all: mpi_matrix_mult

# Compile straight C++ files
#$(OBJ_DIR)/%.o : $(PREFIX)/%.cu
#	$(CC) $(CXXFLAGS) $(CPLUS_INCLUDE_PATH) -c $< -o $@

# Compile straight CUDA files
# -dc avoids error : ptxas fatal   : Unresolved extern function '_Z9d_map_idxiii'
$(OBJ_DIR_MPI)/%.o : $(PREFIX_MPI)/%.cu
	$(NVCC) $(NVCCFLAGS) $(CPLUS_INCLUDE_PATH) -dc $< -o $@

mpi_matrix_mult : $(OBJ_MPI)
	$(NVCC) $(NVCCFLAGS) $(CPLUS_INCLUDE_PATH) -o mpi_matrix_mult $(MPI_LIB) $(OBJ_MPI)

#nccl_matrix_mult : $(OBJ)
#	$(NVCC) $(NVCCFLAGS) $(CPLUS_INCLUDE_PATH) -o mpi_matrix_mult $(LIB) $(OBJ)

clean :
	rm $(OBJ_MPI) mpi_matrix_mult
