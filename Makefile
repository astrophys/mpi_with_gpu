# To compile :
#   ml load openmpi/gcc/64/4.1.2
#
#
LIB = -L/cm/local/apps/cuda/libs/current/lib64 -L/cm/shared/apps/openmpi4/gcc/4.1.2/lib -lmpi
CXXFLAGS = -Wall -g
PREFIX = src
CC = mpic++
NVCC = nvcc
NVCCFLAGS = --compiler-options ${CXXFLAGS} -I${CPATH} -I${CUDA_INC_PATH}/include

OBJ = $(OBJ_DIR)/mpi_matrix_mult.o $(OBJ_DIR)/functions.o $(OBJ_DIR)/cpu_mult.o $(OBJ_DIR)/gpu_mult.o
OBJ_DIR = $(PREFIX)/obj
#CU_OBJ = $(OBJ_DIR)/functions.o

# E.g. of working compilation nvcc -I/cm/shared/apps/openmpi4/gcc/4.1.2/include -I${CUDA_INC_PATH}/include -L/cm/local/apps/cuda/libs/current/lib64 -L/cm/shared/apps/openmpi4/gcc/4.1.2/lib -lmpi -o mpi_matrix_mult src/obj/mpi_matrix_mult.o

all: mpi_matrix_mult

# Compile straight C++ files
#$(OBJ_DIR)/%.o : $(PREFIX)/%.cu
#	$(CC) $(CXXFLAGS) $(CPLUS_INCLUDE_PATH) -c $< -o $@

# Compile straight CUDA files
# -dc avoids error : ptxas fatal   : Unresolved extern function '_Z9d_map_idxiii'
$(OBJ_DIR)/%.o : $(PREFIX)/%.cu
	$(NVCC) $(NVCCFLAGS) $(CPLUS_INCLUDE_PATH) -dc $< -o $@

mpi_matrix_mult : $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(CPLUS_INCLUDE_PATH) -o mpi_matrix_mult $(LIB) $(OBJ)

clean :
	rm $(OBJ) mpi_matrix_mult
