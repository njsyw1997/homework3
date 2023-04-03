# HPC HW3

All the tests are executed on AMD EPYC 7452

Architecture: x86_64

Core: 32 Cores, 64 Threads.

## Problem 1

For convenience, suppose n is an odd number, so n-1 is an even number.

1. In first block

```cpp
Thread 1: 1+2+...(n-1)/2=(n-1)(n+1)/8
Thread 2: (n+1)/2+...+n-1=(n-1)(3n-1)/8
```

It’s similar in the second block. In both blocks, the faster threads need to wait for the slower thread and the execution time is $\frac{3n^2-4n+1}{8}$ for both threads. So for each thread, it takes $\frac{3n^2-4n+1}{4}$ time to execute the code. And it takes $\frac{n^2-2n+1}{4}$ to wait for the other thread.

1. In first block

```cpp
Thread 1: 1+3+...n-2=(n-1)(n-1)/4
Thread 2: 2+4+...n-1=(n+1)(n-1)/4
```

It’s similar in the second block. The faster thread needs to wait for the slower thread. The execution time would change to $\frac{n^2-1}{2}$ if we use schedule(static, 1)

1. No. The execution time would still be $\frac{n^2-1}{2}$
2. Use *#pragma omp for nowait* , so the two threads will execute the code independently without waiting for each other and the execution time would be $\frac{n^2-n}{2}$

## Problem 2

The results are as follows:

| Threads | Time (s) |
| --- | --- |
| Sequential | 0.354813 |
| 1 | 0.150416 |
| 2 | 0.091669 |
| 3 | 0.063784 |
| 4 | 0.056482 |
| 5 | 0.048577 |
| 6 | 0.040894 |
| 7 | 0.047058 |
| 8 | 0.042982 |
| 9 | 0.039339 |
| 10 | 0.044391 |
| 11 | 0.040904 |
| 12 | 0.039311 |
| 13 | 0.037702 |
| 14 | 0.037416 |
| 15 | 0.037534 |
| 16 | 0.036779 |
| 32 | 0.040947 |
| 64 | 0.041079 |

## Problem 3

All tests execute 1000 iteration steps

### Jacobi

| N | 100 | 200 | 300 | 400 | 500 | 1000 | 2000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sequential | 0.0654055 | 0.215951 | 0.482291 | 0.850728 | 1.3304 | 5.54713 | 22.654 |
| 1 | 0.0783834 | 0.304176 | 0.689906 | 1.21919 | 1.90138 | 7.64932 | 30.3746 |
| 2 | 0.0542162 | 0.155366 | 0.348218 | 0.624541 | 0.972101 | 3.90658 | 15.5758 |
| 3 | 0.0423892 | 0.108201 | 0.240097 | 0.421845 | 0.66116 | 2.60396 | 10.4083 |
| 4 | 0.0346767 | 0.0832176 | 0.180066 | 0.321046 | 0.503753 | 1.97442 | 7.94032 |
| 8 | 0.0253706 | 0.047602 | 0.0949416 | 0.163929 | 0.259696 | 1.00539 | 4.06183 |
| 16 | 0.0255214 | 0.0518742 | 0.0635463 | 0.100041 | 0.149779 | 0.547674 | 2.37692 |
| 32 | 0.0292166 | 0.03813 | 0.0524199 | 0.0719498 | 0.102989 | 0.327846 | 1.28452 |

### Gauss-Seidel

| N | 100 | 200 | 300 | 400 | 500 | 1000 | 2000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sequential | 0.064343 | 0.211821 | 0.472488 | 0.84539 | 1.31037 | 5.29642 | 22.1798 |
| 1 | 0.0718781 | 0.284126 | 0.632317 | 1.09958 | 1.68797 | 6.92203 | 27.9554 |
| 2 | 0.0509471 | 0.142195 | 0.310528 | 0.5469 | 0.850348 | 3.45457 | 14.0719 |
| 3 | 0.0405406 | 0.0998542 | 0.22026 | 0.384052 | 0.594392 | 2.34371 | 9.45401 |
| 4 | 0.0352813 | 0.0798004 | 0.166947 | 0.29237 | 0.449853 | 1.77058 | 7.33993 |
| 8 | 0.0307585 | 0.0705555 | 0.1114 | 0.159592 | 0.243039 | 0.926206 | 3.63296 |
| 16 | 0.0297801 | 0.051137 | 0.0711688 | 0.120356 | 0.161601 | 0.560786 | 1.95006 |
| 32 | 0.0409267 | 0.0504803 | 0.0726514 | 0.132253 | 0.182979 | 0.490419 | 1.18584 |