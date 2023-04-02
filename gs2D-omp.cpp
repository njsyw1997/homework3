// Compile: g++ -fopenmp -O3 -g -o gs2D-omp gs2D-omp.cpp
// ./gs2D-omp -n <N> -p <Thread_num>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <iostream>
#include "utils.h"
#include <vector>
#include <cmath>
using namespace std;

double get_value(int N, const vector<double> &x, long i, long j)
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

double norm(int N, const vector<double> &x,const vector<double> &rhs)
{
    double norm_sum=0;
    double h=1/double(N+1);
    double h2=h*h;
    for (long i=0;i< N;i++)
    {
        for (long j = 0; j < N; j++)
        {
            double lb,lu,rb,ru=0; 
            long index=i*N+j;
            lb=get_value(N,x,i-1,j-1);
            lu=get_value(N,x,i-1,j+1);
            rb=get_value(N,x,i+1,j-1);
            ru=get_value(N,x,i+1,j+1);
            norm_sum+=pow((x[index]*4-lb-lu-rb-ru)/h2-rhs[index],2.0);
        } 
        
    }
    return sqrt(norm_sum);     
}



vector<double> seq_gs(long N,  int max_iter, double conv, int &iter, double &residual_norm)
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
                long index=i*N+j; 
                lb=get_value(N,x,i-1,j-1);
                lu=get_value(N,x,i-1,j+1);
                rb=get_value(N,x,i+1,j-1);
                ru=get_value(N,x,i+1,j+1);
                x[index]=(h2*rhs[index]+lb+lu+rb+ru)/4.0; 
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



#if defined(_OPENMP)
double omp_norm(int N, const vector<double> &x,const vector<double> &rhs)
{
    double norm_sum=0;
    double h=1/double(N+1);
    double h2=h*h;
    long N2=N*N;
    #pragma omp parallel for reduction(+:norm_sum) collapse(2)
    for (long i=0;i < N;i++)
        for (long j = 0; j < N; j++)
        {   
            long index=i*N+j;
            double lb,lu,rb,ru=0; 
            lb=get_value(N,x,i-1,j-1);
            lu=get_value(N,x,i-1,j+1);
            rb=get_value(N,x,i+1,j-1);
            ru=get_value(N,x,i+1,j+1);
            norm_sum+=pow((x[index]*4-lb-lu-rb-ru)/h2-rhs[index],2.0);        
        }
    return sqrt(norm_sum);     
}
vector<double> omp_gs(long N,  int max_iter, double conv, int &iter, double &residual_norm)
{   
    bool output=false;
    double h=1/double(N+1);
    double h2=h*h;
    long N2=N*N;
    vector<double> x(N2,0);    
    vector<double> rhs(N2,1);
    double init_res=omp_norm(N,x,rhs);
    double res=10000000;
    int iter_=0;
    for (; (iter_ < max_iter)&&((res/init_res)>conv); iter_++)
    {
        // Scan from left to right,bottom to top      
        #pragma omp parallel for 
        for(long i=0; i < N;i++)
            for (long j = i%2; j < N; j=j+2)               
            {   
                long index=i*N+j;
                double lb,lu,rb,ru=0; 
                lb=get_value(N,x,i-1,j-1);
                lu=get_value(N,x,i-1,j+1);
                rb=get_value(N,x,i+1,j-1);
                ru=get_value(N,x,i+1,j+1);
                x[index]=(h2*rhs[index]+lb+lu+rb+ru)/4.0; 
            } 
        #pragma omp parallel for 
        for(long i=0; i < N;i++)
            for (long j = (i+1)%2; j < N; j=j+2)               
            {   
                long index=i*N+j;
                double lb,lu,rb,ru=0; 
                lb=get_value(N,x,i-1,j-1);
                lu=get_value(N,x,i-1,j+1);
                rb=get_value(N,x,i+1,j-1);
                ru=get_value(N,x,i+1,j+1);
                x[index]=(h2*rhs[index]+lb+lu+rb+ru)/4.0; 
            } 
        res=omp_norm(N,x,rhs);
        if (output)
        {
           cout<<"iter "<<iter_<<" Res="<<res<<endl;   
        }    
    }
    iter=iter_;
    residual_norm=res;
    return x;
}
#endif

int main (int argc, char** argv){
    long n = read_option<long>("-n", argc, argv);
    int p = read_option<long>("-p", argc, argv);
    int seq_iter=0;
    int omp_iter=0;
    double seq_res=0;
    double omp_res=0;

    #if defined(_OPENMP)
    double t=omp_get_wtime();
    vector<double> x1=seq_gs(n,1000,1e-6,seq_iter,seq_res);
    cout<<"Seq iter="<<seq_iter<<" Res="<<seq_res<<" Running time is "<<omp_get_wtime() - t<<endl;
    #else
    Timer t;
    t.tic();
    vector<double> x1=seq_gs(n,1000,1e-6,seq_iter,seq_res);
    cout<<"Seq iter="<<seq_iter<<" Res="<<seq_res<<" Running time is "<<t.toc()<<endl;
    #endif

    #if defined(_OPENMP)
    omp_set_num_threads(p);
    double tt=omp_get_wtime();
    vector<double> x2=omp_gs(n,1000,1e-6,omp_iter,omp_res);    
    cout<<"OMP iter="<<omp_iter<<" Res="<<omp_res<<" Running time is "<<omp_get_wtime() - tt<<endl;
    #endif
    return 0;
}