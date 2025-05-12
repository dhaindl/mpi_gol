#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <mpi.h>
#include <string>

using namespace std;

/*
  Name: Daniel Haindl
  Hw_3: Game of Life MPI
  Date: 5/12/2025
  To Compile: mpic++ -o gol_blocking gol_blocking.cpp
  Run: mpirun -np "number of processes" ./gol_blocking
*/

//method to update the cells between generations
//makes sure to not use the cell in the middle of the 3x3 extracted region
//uses gol rules to check if the next generation's cell will either be dead or alive
void nextGeneration(const vector<int>& current, vector<int>& next, int rows, int cols) {
    for (int i = 1; i <= rows; ++i) {
        for (int j = 1; j <= cols; ++j) {
            int aliveNeighbors = 0;
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {

                    //uses 1D indexing to acces the 2D grid, grid is flattened and padded
                    if (di != 0 || dj != 0) {
                        aliveNeighbors += current[(i + di) * (cols + 2) + (j + dj)];
                    }
                }
            }

            int currentCell = current[i * (cols + 2) + j];
            if (currentCell == 1) {
                next[i * (cols + 2) + j] = (aliveNeighbors == 2 || aliveNeighbors == 3) ? 1 : 0;
            } else {
                next[i * (cols + 2) + j] = (aliveNeighbors == 3) ? 1 : 0;
            }
        }
    }
}

int main(int argc, char** argv) {

    //initialize the MPI execution environment
    MPI_Init(&argc, &argv); 

    //rank: ID of current process
    //size: total number of MPI processes
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //define size of grid
    const int total_rows = 5000;
    const int cols = 5000;
    const int rows = total_rows / size;
    int gen = 5000;
    std::string type = "blocking";

    //allocates the grid, includes ghost cell layers
    vector<int> local((rows + 2) * (cols + 2), 0);  // Includes ghost rows
    vector<int> next((rows + 2) * (cols + 2), 0);

    // Random initialization per process, deterministic by global row
    for (int i = 1; i <= rows; ++i) {
        int global_row = rank * rows + (i - 1);
        srand(global_row); // Consistent seed for row
        for (int j = 1; j <= cols; ++j) {
            local[i * (cols + 2) + j] = rand() % 2;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before timing
    //current wall-clock time
    double start_time = MPI_Wtime();

    //loops over the generations
    //uses circular wrap around for ghost cells
    for (int i = 0; i < gen; ++i) {
        int prev = (rank - 1 + size) % size;
        int nextRank = (rank + 1) % size;

        // Sends the first real row to the previous process, 
        //receives into top ghost row from the next process
        MPI_Sendrecv(&local[(cols + 2)], cols, MPI_INT, prev, 0,
                     &local[0], cols, MPI_INT, nextRank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        //Sends the last real row to the next process, receives 
        //into bottom ghost row from the previous process.
        MPI_Sendrecv(&local[rows * (cols + 2)], cols, MPI_INT, nextRank, 1,
                     &local[(rows + 1) * (cols + 2)], cols, MPI_INT, prev, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        nextGeneration(local, next, rows, cols);
        
        //stops simulation if the grids are equal
        if(local == next){
            break;
        }

        //swaps the two grids to prepare for next generation
        swap(local, next);
    }

    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (rank == 0) {
        ofstream outfile("results.txt", ios::app);
        if (outfile.is_open()) {
            outfile << "Processes: " << size << ", Time: " << elapsed << " seconds" << ", generations: " << gen << ", type: " << type << endl ;
            outfile.close();
        } else {
            cerr << "Failed to open results.txt for writing.\n";
        }
    }

    MPI_Finalize();
    return 0;
}
