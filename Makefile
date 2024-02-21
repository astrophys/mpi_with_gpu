# To compile :
#   ml load openmpi/gcc/64/4.1.2
#
#
LIB =
OPTIONS = -Wall
PREFIX = src
GCC = mpic++

OBJ = $(OBJ_DIR)/mpi_integrate.o 
OBJ_DIR = $(PREFIX)/obj


all: mpi_integrate

$(OBJ_DIR)/%.o : $(PREFIX)/%.cpp
	$(GCC) $(OPTIONS) $(INCLUDE) -c $< -o $@

mpi_integrate : $(OBJ)
	$(GCC) $(OPTIONS) $(INCLUDE) -o mpi_integrate $(LIB) $(OBJ)

clean :
	rm $(OBJ) mpi_integrate
