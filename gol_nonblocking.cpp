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
  To Compile: mpic++ -o gol_nonblocking gol_blocking.cpp
  Run: mpirun -np "number of processes" ./gol_nonblocking
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
    MPI_Init(&argc, &argv); 

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int total_rows = 5000;
    const int cols = 5000;
    const int rows = total_rows / size;
    int gen = 5000;
    std::string type = "non_blocking";

    vector<int> local((rows + 2) * (cols + 2), 0);  // Includes ghost rows
    vector<int> next((rows + 2) * (cols + 2), 0);

    // Deterministic random init per row
    for (int i = 1; i <= rows; ++i) {
        int global_row = rank * rows + (i - 1);
        srand(global_row);
        for (int j = 1; j <= cols; ++j) {
            local[i * (cols + 2) + j] = rand() % 2;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // Sync before timing
    double start_time = MPI_Wtime();

    for (int i = 0; i < gen; ++i) {
        int prev = (rank - 1 + size) % size;
        int nextRank = (rank + 1) % size;

        MPI_Request requests[4];

        // Post non-blocking receives
        MPI_Irecv(&local[0], cols, MPI_INT, nextRank, 0, MPI_COMM_WORLD, &requests[0]); // top ghost row
        MPI_Irecv(&local[(rows + 1) * (cols + 2)], cols, MPI_INT, prev, 1, MPI_COMM_WORLD, &requests[1]); // bottom ghost row

        // Post non-blocking sends
        MPI_Isend(&local[(cols + 2)], cols, MPI_INT, prev, 0, MPI_COMM_WORLD, &requests[2]); // send top row
        MPI_Isend(&local[rows * (cols + 2)], cols, MPI_INT, nextRank, 1, MPI_COMM_WORLD, &requests[3]); // send bottom row

        // Wait for all communication to complete
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

        // Compute next generation
        nextGeneration(local, next, rows, cols);

        if (local == next) break; // Optional early stop if converged
        swap(local, next);
    }

    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (rank == 0) {
        ofstream outfile("results.txt", ios::app);
        if (outfile.is_open()) {
            outfile << "Processes: " << size << ", Time: " << elapsed << " seconds, Generations: " << gen << ", type:  " << type <<endl;
            outfile.close();
        } else {
            cerr << "Failed to open results.txt for writing.\n";
        }
    }

    MPI_Finalize();
    return 0;
}