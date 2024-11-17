# Compiler and flags
CXX = mpic++
CXXFLAGS = -std=c++11

# Target executable name
TARGET = array_average

# Source file
SRC = array_average.cpp

# Number of processes (default to 5 if not specified)
NP ?= 5

# Host file
HOSTFILE = mpi_host

# Default target to build the executable
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

# Run the program with a configurable number of processes
run:
	mpirun -np $(NP) -f $(HOSTFILE) ./$(TARGET)

# Clean the build files
clean:
	rm -f $(TARGET)

