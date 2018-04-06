// Author: Sameer Deshmukh (@v0dro)
// Desc: A simple program for multiplying two matrices using PBLAS pdgemm.

#include "mpi.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
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

  void pdgemm_( char* TRANSA, char* TRANSB,
                int * M, int * N, int * K,
                double * ALPHA,
                double * A, int * IA, int * JA, int * DESCA,
                double * B, int * IB, int * JB, int * DESCB,
                double * BETA,
                double * C, int * IC, int * JC, int * DESCC );
}


int main(int argc, char ** argv)
{
  // MPI init
  MPI_Init(&argc, &argv);
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  // end MPI init

  // BLACS init
  int BLACS_CONTEXT, proc_nrows, proc_ncols, myrow, mycol;
  int proc_id, num_procs;
  proc_nrows = 2; proc_ncols = 2;
  //int proc_dims[2] = {proc_nrows, proc_ncols};
  Cblacs_pinfo(&proc_id, &num_procs);
  Cblacs_get( -1, 0, &BLACS_CONTEXT );
  Cblacs_gridinit( &BLACS_CONTEXT, "Row", proc_nrows, proc_ncols );
  Cblacs_pcoord(BLACS_CONTEXT, mpi_rank, &myrow, &mycol);
  cout << "myrow " << myrow << " mycol " << mycol << endl;
  cout << "procid " << proc_id << " num_procs " << num_procs << endl;
  // end BLACS init

  // matrix properties
  int N = 8, nb = 4; // mat size, blk size.
  double* a = (double*)malloc(sizeof(double)*nb*nb);
  double* b = (double*)malloc(sizeof(double)*nb*nb);
  double* c = (double*)malloc(sizeof(double)*nb*nb);

  // generate matrix data
  for (int i = 0; i < nb*nb; ++i) {
    a[i] = 1;
    b[i] = 2;
    c[i] = 0;
  }
  // end matrix properties

  // create array descriptor
  int desca[9];
  int descb[9];
  int descc[9];
  int rsrc = 0, csrc = 0, info;
  descinit_(desca, &N, &N, &nb, &nb, &rsrc, &csrc, &BLACS_CONTEXT, &nb, &info);
  descinit_(descb, &N, &N, &nb, &nb, &rsrc, &csrc, &BLACS_CONTEXT, &nb, &info);
  descinit_(descc, &N, &N, &nb, &nb, &rsrc, &csrc, &BLACS_CONTEXT, &nb, &info);
  cout << proc_id << " info: " << info << endl;
  // end create array descriptor
  
  Cblacs_barrier(BLACS_CONTEXT, "All");
  int ia = 1, ja = 1, ib = 1, jb = 1, ic = 1, jc = 1;
  double alpha = 1, beta = 1;
  pdgemm_("T", "T", &N, &N, &N, &alpha, a, &ia, &ja, desca, b, &ib, &jb, descb,
          &beta, c, &ic, &jc, descc);

  // print results on a per-process basis
  if (proc_id == 0) {
    cout << "proc : " << proc_id << endl;
    for (int i = 0; i < nb; ++i) {
      for (int j = 0; j < nb; ++j) {
        cout << "(" << nb*myrow + i << "," <<
          nb*mycol + j << ") " << c[i*nb + j] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }

  if (proc_id == 1) {
    cout << "proc : " << proc_id << endl;
    for (int i = 0; i < nb; ++i) {
      for (int j = 0; j < nb; ++j) {
        cout << "(" << nb*myrow + i << "," <<
          nb*mycol + j << ") " << c[i*nb + j] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }

  if (proc_id == 2) {
    cout << "proc : " << proc_id << endl;
    for (int i = 0; i < nb; ++i) {
      for (int j = 0; j < nb; ++j) {
        cout << "(" << nb*myrow + i << "," <<
          nb*mycol + j << ") " << c[i*nb + j] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }

  if (proc_id == 3) {
    cout << "proc : " << proc_id << endl;
    for (int i = 0; i < nb; ++i) {
      for (int j = 0; j < nb; ++j) {
        cout << "(" << nb*myrow + i << "," <<
          nb*mycol + j << ") " << c[i*nb + j] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }

  MPI_Finalize();
}
