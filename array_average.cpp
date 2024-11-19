#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <numeric>

using namespace std;

// Function to generate an array of random numbers between 0 and 99
vector<int> generateArray(int size) {
    vector<int> arr(size);
    srand(time(0));
    for (int i = 0; i < size; ++i) {
        arr[i] = rand() % 100; 
    }
    return arr;
}

// Function to calculate the average of a vector
double calculateAverage(const vector<int>& subarray) {
    double sum = 0;
    for (int num : subarray) {
        sum += num;
    }
    return sum / subarray.size();
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    const int array_size = 1000;
    vector<int> array;

    if (world_rank == 0) {
        array = generateArray(array_size);
    }

    // Determine the number of elements for each process
    vector<int> send_counts(world_size, array_size / world_size);
    vector<int> displacements(world_size, 0);

    // Handle uneven division by adding remaining elements to the first processes
    for (int i = 0; i < array_size % world_size; ++i) {
        send_counts[i]++;
    }

    for (int i = 1; i < world_size; ++i) {
        displacements[i] = displacements[i - 1] + send_counts[i - 1];
    }

    vector<int> subarray(send_counts[world_rank]);

    // Root process sends data to other processes
    if (world_rank == 0) {
        for (int i = 1; i < world_size; ++i) {
            MPI_Send(
                array.data() + displacements[i], // data
                send_counts[i],                  // count
                MPI_INT,                         // datatype
                i,                               // destination
                0,                               // Tag
                MPI_COMM_WORLD                   // Communicator
            );
        }
        // Root process extracts its subarray
        subarray.assign(array.begin(), array.begin() + send_counts[0]);
    } else {
        // Other processes receive their subarray
        MPI_Recv(
            subarray.data(),                 // data
            send_counts[world_rank],         // count
            MPI_INT,                         // datatype
            0,                               // Source 
            0,                               // Tag
            MPI_COMM_WORLD,                  // Communicator
            MPI_STATUS_IGNORE                // status
        );
    }

    // Calculate the average of the subarray
    double subarray_avg = calculateAverage(subarray);

    // Gather subarray averages at the root process
    vector<double> all_averages(world_size);
    MPI_Gather(
        &subarray_avg,
        1,
        MPI_DOUBLE,
        all_averages.data(),
        1,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // Gather local outputs at the root process
    string local_output = "Process " + to_string(world_rank) +
                          " on processor " + string(processor_name) +
                          " received " + to_string(subarray.size()) +
                          " elements: ";

    for (int num : subarray) {
        local_output += to_string(num) + " ";
    }

    local_output += "\nSubarray average: " + to_string(subarray_avg) + "\n";

    int local_size = local_output.size();
    vector<int> sizes(world_size);
    MPI_Gather(
        &local_size,
        1,
        MPI_INT,
        sizes.data(),
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    vector<int> displs(world_size, 0);
    if (world_rank == 0) {
        for (int i = 1; i < world_size; ++i) {
            displs[i] = displs[i - 1] + sizes[i - 1];
        }
    }

    int total_size = accumulate(sizes.begin(), sizes.end(), 0);
    vector<char> gathered_output(total_size);
    MPI_Gatherv(
        local_output.data(),
        local_size,
        MPI_CHAR,
        gathered_output.data(),
        sizes.data(),
        displs.data(),
        MPI_CHAR,
        0,
        MPI_COMM_WORLD
    );

    if (world_rank == 0) {
        cout << "*************************************\n";
        cout << "Collected subarrays from all processes:\n";
        cout << string(gathered_output.begin(), gathered_output.end());
        cout << "*************************************\n";

        // Calculate overall average
        double overall_sum = 0;
        for (int i = 0; i < world_size; ++i) {
            cout << "Process " << i << " average: " << all_averages[i]
                 << " (count: " << send_counts[i] << ")\n";
            overall_sum += all_averages[i] * send_counts[i];
        }
        
        double total_avg = overall_sum / array_size;
        cout << "Overall average of the array: " << total_avg << endl;
        cout << "*************************************\n";
    }

    MPI_Finalize();
    return 0;
}
