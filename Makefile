# To compile :
#   ml load openmpi/gcc/64/4.1.2
#
#
LIB = -L/cm/local/apps/cuda/libs/current/lib64 -L/cm/shared/apps/openmpi4/gcc/4.1.2/lib -lmpi
CXXFLAGS = -Wall
PREFIX = src
CC = mpic++
NVCC = nvcc
NVCCFLAGS = --compiler-options ${CXXFLAGS} -I${CPATH} -I${CUDA_INC_PATH}/include

OBJ = $(OBJ_DIR)/mpi_communicate.o $(OBJ_DIR)/functions.o
OBJ_DIR = $(PREFIX)/obj
#CU_OBJ = $(OBJ_DIR)/functions.o

# E.g. of working compilation nvcc -I/cm/shared/apps/openmpi4/gcc/4.1.2/include -I${CUDA_INC_PATH}/include -L/cm/local/apps/cuda/libs/current/lib64 -L/cm/shared/apps/openmpi4/gcc/4.1.2/lib -lmpi -o mpi_communicate src/obj/mpi_communicate.o

all: mpi_communicate

# Compile straight C++ files
#$(OBJ_DIR)/%.o : $(PREFIX)/%.cu
#	$(CC) $(CXXFLAGS) $(CPLUS_INCLUDE_PATH) -c $< -o $@

# Compile straight CUDA files
$(OBJ_DIR)/%.o : $(PREFIX)/%.cu
	$(NVCC) $(NVCCFLAGS) $(CPLUS_INCLUDE_PATH) -c $< -o $@

mpi_communicate : $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(CPLUS_INCLUDE_PATH) -o mpi_communicate $(LIB) $(OBJ)

clean :
	rm $(OBJ) mpi_communicate
