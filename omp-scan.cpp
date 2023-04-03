// Compile: g++ -fopenmp -O3 -g -o omp-scan omp-scan.cpp
// Execute: ./omp-scan
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

#if defined(_OPENMP)
void scan_omp(long* prefix_sum, const long* A, long n) {
  // Suppose n is bigger than the number of threads
  if (n == 0) return;
  long* r;
  #pragma omp parallel shared(r)
  { 
    int p = omp_get_num_threads();
    int t = omp_get_thread_num();
    # pragma omp single
    {
      r= (long*) malloc(p * sizeof(long));
      // printf("Number of threads = %d\n",p);
    }
    // Fill out parallel scan: One way to do this is array into p chunks
    // Do a scan in parallel on each chunk, then share/compute the offset
    // through a shared vector and update each chunk by adding the offset
    // in parallel
    long start_index = t * (n-1) / p +1;
    long end_index = (t + 1) * (n-1) / p+1;
    prefix_sum[start_index] = A[start_index-1];
    for (long i = start_index+1; i < end_index; i++) {      
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    }    
    r[t]=prefix_sum[end_index-1];
    #pragma omp barrier
    long r_sum=0;
    for (size_t i = 0; i < t; i++)
    {
      r_sum += r[i];
    }
    for (long j = start_index; j < end_index; j++)
    {
      prefix_sum[j] += r_sum;
    } 
  }
  free(r);
}
#endif


int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N; i++) B1[i] = 0;

#if defined(_OPENMP)
  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);
#else
  Timer t;
  t.tic();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", t.toc());
#endif


#if defined(_OPENMP)
  for (size_t threads = 1; threads <= 64; threads=threads*2)
  {
    omp_set_num_threads(threads);
    tt = omp_get_wtime();
    scan_omp(B1, A, N);
    printf("parallel-scan with %d threads   = %fs",threads, omp_get_wtime() - tt);
    long err = 0;
    for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
    printf("  error = %ld\n", err);
  } 
#endif
  free(A);
  free(B0);
  free(B1);
  return 0;
}
