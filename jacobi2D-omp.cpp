// Compile: g++ -fopenmp -O3 -g -o jacobi2D-omp jacobi2D-omp.cpp
#ifdef(_OPENMP)
#include <omp.h>
#endif
#include <iostream>
#include "utils.h"
#include <vector>
#include <cmath>
using namespace std;


double norm(int N, const vector<double> &x,const vector<double> &rhs)
{
    double norm_sum=0;
    double h=1/double(N+1);
    double h2=h*h;
    for (long i = 0; i < N; i++)
    {
        if (i==0)
        {
            norm_sum+=pow((x[i]*2/h2-x[i+1]/h2-rhs[i]),2.0);
        }
        if (i==N-1)
        {
            norm_sum+=pow((x[i]*2/h2-x[i-1]/h2-rhs[i]),2.0);
            continue;
        }   
        norm_sum+=pow((x[i]*2/h2-x[i-1]/h2-x[i+1]/h2-rhs[i]),2.0);
    }
    return sqrt(norm_sum);     
}

double get_value(int N, vector<double> &x, long i, long j)
{
    if (i<0)
    {
        return 0;
    }
    if (i>=N)
    {
        return 0;
    }
    if (j<0)
    {
        return 0;
    }
    if (j>=N)
    {
        return 0;
    }
    return x[i*N+j];
}

vector seq_jacobi(long N,  int max_iter, double conv, int &iter, double &residual_norm)
{   
    bool output=false;
    double h=1/double(N+1);
    double h2=h*h;
    long N2=N*N;
    vector<double> x(N2,0);    
    vector<double> prev_x(x);
    vector<double> rhs(N2,1);
    double init_res=norm(N,x,rhs);
    double res=10000000;
    int iter_=0;
    for (; (iter_ < max_iter)&&((res/init_res)>conv); iter_++)
    {
        // Scan from left to right,bottom to top
        for (long i=0;i< N;i++)
        {
            for (long j = 0; j < N; j++)
            {
                double lb,lu,rb,ru=0; 
                lb=get_value(N,prev_x,i-1,j-1);
                lu=get_value(N,prev_x,i-1,j+1);
                rb=get_value(N,prev_x,i+1,j-1);
                ru=get_value(N,prev_x,i+1,j+1);
                x[j]=(h2*rhs[j]+lb+lu+rb+ru)/4.0; 
            }
        }    
        res=norm(N,x,rhs);
        if (output)
        {
           cout<<"iter "<<iter_<<" Res="<<res<<endl;   
        }   
        prev_x=x;    
    }
    iter=iter_;
    residual_norm=res;
    return x;
}

vector seq_gauss_seidel(long N,  int max_iter, double conv, int &iter, double &residual_norm)
{   
    bool output=false;
    double h=1/double(N+1);
    double h2=h*h;
    long N2=N*N;
    vector<double> x(N2,0);    
    vector<double> rhs(N2,1);
    double init_res=norm(N,x,rhs);
    double res=10000000;
    int iter_=0;
    for (; (iter_ < max_iter)&&((res/init_res)>conv); iter_++)
    {
        // Scan from left to right,bottom to top
        for (long i=0;i< N;i++)
        {
            for (long j = 0; j < N; j++)
            {
                double lb,lu,rb,ru=0; 
                lb=get_value(N,x,i-1,j-1);
                lu=get_value(N,x,i-1,j+1);
                rb=get_value(N,x,i+1,j-1);
                ru=get_value(N,x,i+1,j+1);
                x[j]=(h2*rhs[j]+lb+lu+rb+ru)/4.0; 
            }
        }    
        res=norm(N,x,rhs);
        if (output)
        {
           cout<<"iter "<<iter_<<" Res="<<res<<endl;   
        }    
    }
    iter=iter_;
    residual_norm=res;
    return x;
}

#ifdef

int main (int argc, char** argv){
    long n = read_option<long>("-n", argc, argv);
    int jac_iter=0;
    int gs_iter=0;
    Timer t;
    double gs_time=0,jac_time=0;
    double jac_norm=0;
    double gs_norm=0;
    cout<<"Jacobi Iterations begin"<<endl;
    t.tic();
    jacobi(n,100,1e-4,jac_iter,jac_norm);
    jac_time=t.toc();
    cout<<"Gauss-Seidel Iterations begin"<<endl;
    t.tic();
    gauss_seidel(n,100,1e-4,gs_iter,gs_norm);
    gs_time=t.toc();
    cout<<"Jacobi iterations are "<<jac_iter<<", residual norm is "<<jac_norm<<", running time is "<<jac_time<<endl;
    cout<<"Gauss-Seidel iterations are "<<gs_iter<<", residual norm is "<<gs_norm<<", running time is "<<gs_time<<endl;
    return 0;
}