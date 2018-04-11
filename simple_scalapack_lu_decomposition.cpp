// Implement simple distributed LU decomposition using scalapack.
// author: Sameer Deshmukh (@v0dro)

#include "mpi.h"
#include <cstdlib>
#include <cmath>
#include <iostream>
using namespace std;

extern "C" {
  /* Cblacs declarations */
  void Cblacs_pinfo(int*, int*);
  void Cblacs_get(int, int, int*);
  void Cblacs_gridinit(int*, const char*, int, int);
  void Cblacs_pcoord(int, int, int*, int*);
  void Cblacs_gridexit(int);
  void Cblacs_barrier(int, const char*);
 
  int numroc_(int*, int*, int*, int*, int*);

  void descinit_(int *desc, const int *m,  const int *n, const int *mb, 
    const int *nb, const int *irsrc, const int *icsrc, const int *ictxt, 
    const int *lld, int *info);
  void pdgetrf_(
                int *m, int *n, double *a, int *ia, int *ja, int *desca,
                int *ipiv,int *info);
}

int main(int argc, char ** argv)
{
  // MPI init
  MPI_Init(&argc, &argv);
  // end MPI Init
  
  // BLACS init
  int BLACS_CONTEXT, proc_nrows, proc_ncols, myrow, mycol;
  int proc_id, num_procs;
  proc_nrows = 2; proc_ncols = 2;
  Cblacs_pinfo(&proc_id, &num_procs);
  Cblacs_get( -1, 0, &BLACS_CONTEXT );
  Cblacs_gridinit( &BLACS_CONTEXT, "Row", proc_nrows, proc_ncols );
  Cblacs_pcoord(BLACS_CONTEXT, proc_id, &myrow, &mycol);
  // end BLACS init

  // matrix properties
  // mat size, blk size, portion of block per process
  int N = 8, nb = 4, process_block_size = 2;
  int num_blocks_per_process = N/process_block_size;
  int block_size_per_process_r = sqrt(num_blocks_per_process);
  int block_size_per_process_c = sqrt(num_blocks_per_process);
  double* a = (double*)malloc(sizeof(double)*nb*nb);
  // generate matrix data
  for (int j = 0; j < nb; ++j) {
    for (int i = 0; i < nb; ++i) {
      int index = i*nb + j;
      int row_i = myrow*nb + i;
      int col_j = mycol*nb + j;
      a[index] = row_i + col_j;
    }
    cout << endl;
  }
  // end matrix properties

  // print input matrix
  Cblacs_barrier(BLACS_CONTEXT, "All");
  if (proc_id == 0) {
    cout << "proc_id: " << proc_id << endl;
    for (int i = 0; i < nb; ++i) {
      for (int j = 0; j < nb; ++j) {
        cout << a[i*nb + j] << " ";
      }
      cout << endl;
    }
  }
  Cblacs_barrier(BLACS_CONTEXT, "All");
  if (proc_id == 1) {
    cout << "proc_id: " << proc_id << endl;
    for (int i = 0; i < nb; ++i) {
      for (int j = 0; j < nb; ++j) {
        cout << a[i*nb + j] << " ";
      }
      cout << endl;
    }
  }
  Cblacs_barrier(BLACS_CONTEXT, "All");
  if (proc_id == 2) {
    cout << "proc_id: " << proc_id << endl;
    for (int i = 0; i < nb; ++i) {
      for (int j = 0; j < nb; ++j) {
        cout << a[i*nb + j] << " ";
      }
      cout << endl;
    }
  }
  Cblacs_barrier(BLACS_CONTEXT, "All");
  if (proc_id == 3) {
    cout << "proc_id: " << proc_id << endl;
    for (int i = 0; i < nb; ++i) {
      for (int j = 0; j < nb; ++j) {
        cout << a[i*nb + j] << " ";
      }
      cout << endl;
    }
  }
  // end print input matrix

  // create array descriptor
  int desca[9];
  int rsrc = 0, csrc = 0, info;
  descinit_(desca, &N, &N, &nb, &nb, &rsrc, &csrc, &BLACS_CONTEXT, &nb, &info);
  cout << proc_id << " info: " << info << endl;
  // end create array descriptor

  Cblacs_barrier(BLACS_CONTEXT, "All");

  // LU decomposition
  int ia = 1, ja = 1;
  int *ipiv = (int*)malloc(sizeof(int)*N);
  pdgetrf_(&N, &N, a, &ia, &ja, desca, ipiv, &info);
  // end LU decomposition

  // print output matrix
  Cblacs_barrier(BLACS_CONTEXT, "All");
  if (proc_id == 0) {
    cout << "proc_id: " << proc_id << endl;
    for (int j = 0; j < nb; ++j) {
      for (int i = 0; i < nb; ++i) {
        cout << a[i*nb + j] << " ";
      }
      cout << endl;
    }
  }
  Cblacs_barrier(BLACS_CONTEXT, "All");
  if (proc_id == 1) {
    cout << "proc_id: " << proc_id << endl;
    for (int j = 0; j < nb; ++j) {
      for (int i = 0; i < nb; ++i) {
        cout << a[i*nb + j] << " ";
      }
      cout << endl;
    }
  }
  Cblacs_barrier(BLACS_CONTEXT, "All");
  if (proc_id == 2) {
    cout << "proc_id: " << proc_id << endl;
    for (int j = 0; j < nb; ++j) {
      for (int i = 0; i < nb; ++i) {
        cout << a[i*nb + j] << " ";
      }
      cout << endl;
    }
  }
  Cblacs_barrier(BLACS_CONTEXT, "All");
  if (proc_id == 3) {
    cout << "proc_id: " << proc_id << endl;
    for (int j = 0; j < nb; ++j) {
      for (int i = 0; i < nb; ++i) {
        cout << a[i*nb + j] << " ";
      }
      cout << endl;
    }
  }
  // end print output matrix
  MPI_Finalize();
}
