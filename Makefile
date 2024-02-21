# To compile :
#   ml load openmpi/gcc/64/4.1.2
#
#
LIB =
OPTIONS = -Wall
PREFIX = src
GCC = mpicc
OBJ = mpi_integrate.o cpu_mult.c


all: mpi_integrate

$(OBJ_DIR)/%.o : $(PREFIX)/%.c
	$(GCC) $(OPTIONS) $(INCLUDE) -c $< -o $@

mpi_integrate: $(OBJ)
	$(GCC) $(OPTIONS) $(INCLUDE) -o mpi_integrate $(LIB)

clean :
	rm $(OBJ) mpi_integrate
