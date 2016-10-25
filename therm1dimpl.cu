#include <iostream>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#define EPS 1e-3
//#define WRITE_TO_FILE

using namespace std;

//Обработчик ошибок
static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR( error ) (HandleError( error, __FILE__, __LINE__ ))

__global__ void iter_kernel_shared(double *U,double *Unext,double *M,double *F,int Nn,double* err)
{
    int tid=threadIdx.x+blockIdx.x*blockDim.x+1;
    int stid=threadIdx.x+1;
    __shared__ double Utemp[1026];
    if(tid<Nn+1)
    {
        Utemp[stid]=U[tid];

        if(stid==1)
            Utemp[0]=U[tid-1];
        else if(stid==1024)
            Utemp[1025]=U[tid+1];

        __syncthreads();

        double unext;
        unext=Utemp[stid]+1.0/M[(tid-1)*3+1]*(F[tid]-M[(tid-1)*3]*Utemp[stid-1]-M[(tid-1)*3+1]*Utemp[stid]-M[(tid-1)*3+2]*Utemp[stid+1]);
        Unext[tid]=unext;
        err[tid]=abs(unext-Utemp[stid]);
    }
}

__global__ void iter_kernel(double *U,double *Unext,double *M,double *F,int Nn,double* err)
{
    int tid=threadIdx.x+blockIdx.x*blockDim.x+1;
    if(tid<Nn+1)
    {
        double unext;
        unext=U[tid]+1.0/M[(tid-1)*3+1]*(F[tid]-M[(tid-1)*3]*U[tid-1]-M[(tid-1)*3+1]*U[tid]-M[(tid-1)*3+2]*U[tid+1]);
        Unext[tid]=unext;
        err[tid]=abs(unext-U[tid]);
    }
}


float solveGPUshared(double L, double T, double tau, int N)
{
#ifdef WRITE_TO_FILE
    ofstream ofile("../datagpu.dat");
    ofile.precision(16);
    int counter=0, writeeach=1;
#endif
    cudaEvent_t start,stop;
    float gputime=0;
    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    double *U,*Unext,*Uloc;
    double *M,*Mdev,*Fdev,*errdev;
    int Nn=N+1;
    int Nplus=Nn+2;
    double h=L/N,t=0.0;
    size_t size=Nplus*sizeof(double);
    size_t sizeM=3*Nn*sizeof(double);
    Uloc=new double[Nplus];
    M=new double[Nn*3];
    double maxerr;
    HANDLE_ERROR( cudaMalloc(&U,size) );
    HANDLE_ERROR( cudaMalloc(&Unext,size) );
    HANDLE_ERROR( cudaMalloc(&Mdev,sizeM) );
    HANDLE_ERROR( cudaMalloc(&Fdev,size) );
    HANDLE_ERROR( cudaMalloc(&errdev,size) );
    thrust::device_ptr<double> err_ptr = thrust::device_pointer_cast(errdev);
    M[0]=0.0;
    M[1]=1.0;
    M[2]=0.0;
    for(int i=1;i<Nn-1;i++)
    {
        M[i*3]=-tau/(h*h);
        M[i*3+1]=1.0+2.0*tau/(h*h);
        M[i*3+2]=-tau/(h*h);
    }
    M[(Nn-1)*3]=-2.0*tau/(h*h);
    M[(Nn-1)*3+1]=1.0+2.0*tau/(h*h);
    M[(Nn-1)*3+2]=0.0;
    HANDLE_ERROR( cudaMemcpy(Mdev,M,sizeM,cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemset(U,0,size) );
    HANDLE_ERROR( cudaMemset(Fdev,0,size) );
    memset(Uloc,0,size);
    dim3 threads(1024,1,1),blocks(Nn%1024==0?Nn/1024:Nn/1024+1,1,1);
    HANDLE_ERROR( cudaEventRecord(start) );
    while(t<T-0.5*tau)
    {
        HANDLE_ERROR( cudaMemcpy(Fdev,U,size,cudaMemcpyDeviceToDevice) );
        double a=0.0;
        double b=5.0*2.0*tau/h+Uloc[N+1];
        HANDLE_ERROR( cudaMemcpy(&Fdev[1],&a,sizeof(double),cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(&Fdev[N+1],&b,sizeof(double),cudaMemcpyHostToDevice) );
        do{
                iter_kernel_shared<<<blocks,threads>>>(U,Unext,Mdev,Fdev,Nn,errdev);
                HANDLE_ERROR( cudaGetLastError() );
                HANDLE_ERROR( cudaDeviceSynchronize() );
                thrust::device_ptr<double> max_ptr = thrust::max_element(err_ptr+1, err_ptr + Nn+1);
                maxerr=max_ptr[0];
                swap(U,Unext);
        }while(maxerr>EPS);
        t+=tau;
#ifdef WRITE_TO_FILE
        if(counter%writeeach==0)
        {
            HANDLE_ERROR( cudaMemcpy(Uloc,U,size,cudaMemcpyDeviceToHost) );
            for(int i=0;i<Nn;i++)
            ofile<<Uloc[i+1]<<endl;
            ofile<<endl;
            ofile<<endl;
        }
        counter++;
#endif
    }
    HANDLE_ERROR( cudaMemcpy(Uloc,U,size,cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaEventRecord(stop) );
    HANDLE_ERROR( cudaEventSynchronize(stop) );
    HANDLE_ERROR( cudaEventElapsedTime(&gputime,start,stop) );
#ifdef WRITE_TO_FILE
    ofile.close();
#endif

    delete[] Uloc;
    delete[] M;
    HANDLE_ERROR( cudaFree(U) );
    HANDLE_ERROR( cudaFree(Unext) );
    HANDLE_ERROR( cudaFree(Mdev) );
    HANDLE_ERROR( cudaFree(Fdev) );
    HANDLE_ERROR( cudaFree(errdev) );
    HANDLE_ERROR( cudaEventDestroy(start) );
    HANDLE_ERROR( cudaEventDestroy(stop) );
    return 1e-3*gputime;
}

float solveGPU(double L, double T, double tau, int N)
{
#ifdef WRITE_TO_FILE
    ofstream ofile("../datagpu.dat");
    ofile.precision(16);
    int counter=0, writeeach=1;
#endif
    cudaEvent_t start,stop;
    float gputime=0;
    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    double *U,*Unext,*Uloc;
    double *M,*Mdev,*Fdev,*errdev;
    int Nn=N+1;
    int Nplus=Nn+2;
    double h=L/N,t=0.0;
    size_t size=Nplus*sizeof(double);
    size_t sizeM=3*Nn*sizeof(double);
    Uloc=new double[Nplus];
    M=new double[Nn*3];
    double maxerr;
    HANDLE_ERROR( cudaMalloc(&U,size) );
    HANDLE_ERROR( cudaMalloc(&Unext,size) );
    HANDLE_ERROR( cudaMalloc(&Mdev,sizeM) );
    HANDLE_ERROR( cudaMalloc(&Fdev,size) );
    HANDLE_ERROR( cudaMalloc(&errdev,size) );
    thrust::device_ptr<double> err_ptr = thrust::device_pointer_cast(errdev);
    M[0]=0.0;
    M[1]=1.0;
    M[2]=0.0;
    for(int i=1;i<Nn-1;i++)
    {
        M[i*3]=-tau/(h*h);
        M[i*3+1]=1.0+2.0*tau/(h*h);
        M[i*3+2]=-tau/(h*h);
    }
    M[(Nn-1)*3]=-2.0*tau/(h*h);
    M[(Nn-1)*3+1]=1.0+2.0*tau/(h*h);
    M[(Nn-1)*3+2]=0.0;
    HANDLE_ERROR( cudaMemcpy(Mdev,M,sizeM,cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemset(U,0,size) );
    memset(Uloc,0,size);
    dim3 threads(1024,1,1),blocks(Nn%1024==0?Nn/1024:Nn/1024+1,1,1);
    HANDLE_ERROR( cudaEventRecord(start) );
    while(t<T-0.5*tau)
    {
        HANDLE_ERROR( cudaMemcpy(Fdev,U,size,cudaMemcpyDeviceToDevice) );
        double a=0.0;
        double b=5.0*2.0*tau/h+Uloc[N+1];
        HANDLE_ERROR( cudaMemcpy(&Fdev[1],&a,sizeof(double),cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(&Fdev[N+1],&b,sizeof(double),cudaMemcpyHostToDevice) );
        do{
                iter_kernel<<<blocks,threads>>>(U,Unext,Mdev,Fdev,Nn,errdev);
                HANDLE_ERROR( cudaGetLastError() );
                HANDLE_ERROR( cudaDeviceSynchronize() );
                thrust::device_ptr<double> max_ptr = thrust::max_element(err_ptr+1, err_ptr + Nn+1);
                maxerr=max_ptr[0];
                swap(U,Unext);
        }while(maxerr>EPS);

        t+=tau;
#ifdef WRITE_TO_FILE
        if(counter%writeeach==0)
        {
            HANDLE_ERROR( cudaMemcpy(Uloc,U,size,cudaMemcpyDeviceToHost) );
            for(int i=0;i<Nn;i++)
            ofile<<Uloc[i+1]<<endl;
            ofile<<endl;
            ofile<<endl;
        }
        counter++;
#endif
    }
    HANDLE_ERROR( cudaMemcpy(Uloc,U,size,cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaEventRecord(stop) );
    HANDLE_ERROR( cudaEventSynchronize(stop) );
    HANDLE_ERROR( cudaEventElapsedTime(&gputime,start,stop) );
#ifdef WRITE_TO_FILE
    ofile.close();
#endif

    delete[] Uloc;
    delete[] M;
    HANDLE_ERROR( cudaFree(U) );
    HANDLE_ERROR( cudaFree(Unext) );
    HANDLE_ERROR( cudaFree(Mdev) );
    HANDLE_ERROR( cudaFree(Fdev) );
    HANDLE_ERROR( cudaFree(errdev) );
    HANDLE_ERROR( cudaEventDestroy(start) );
    HANDLE_ERROR( cudaEventDestroy(stop) );
    return 1e-3*gputime;
}

float solveCPU(double L, double T, double tau, int N)
{
#ifdef WRITE_TO_FILE
    ofstream ofile("../datacpu.dat");
    ofile.precision(16);
    int counter=0, writeeach=1;
#endif
    float cputime=0;
    double *U,*Unext,*F;
    double *M;
    int Nn=N+1;
    double h=L/N,t=0.0;
    F=new double[Nn];
    U=new double[Nn];
    Unext=new double[Nn];
    M=new double[Nn*3];
    double maxerr;
    M[0]=0.0;
    M[1]=1.0;
    M[2]=0.0;
    for(int i=1;i<Nn-1;i++)
    {
        M[i*3]=-tau/(h*h);
        M[i*3+1]=1.0+2.0*tau/(h*h);
        M[i*3+2]=-tau/(h*h);
    }
    M[(Nn-1)*3]=-2.0*tau/(h*h);
    M[(Nn-1)*3+1]=1.0+2.0*tau/(h*h);
    M[(Nn-1)*3+2]=0.0;
    memset(U,0,Nn*sizeof(double));
    cputime=clock();
    while(t<T-0.5*tau)
    {
        F[0]=0.0;
        F[N]=5.0*2.0*tau/h+U[N];
        for(int i=1;i<Nn-1;i++)
                F[i]=U[i];
        do{
            maxerr=0;
            Unext[0]=U[0]+1.0/M[1]*(F[0]-M[1]*U[0]-M[2]*U[1]);
            for(int i=1;i<Nn-1;i++)
                Unext[i]=U[i]+1.0/M[i*3+1]*(F[i]-M[i*3]*U[i-1]-M[i*3+1]*U[i]-M[i*3+2]*U[i+1]);
            Unext[Nn-1]=U[Nn-1]+1.0/M[(Nn-1)*3+1]*(F[Nn-1]-M[(Nn-1)*3]*U[Nn-2]-M[(Nn-1)*3+1]*U[Nn-1]);
            for(int i=0;i<Nn;i++)
            {
                double err=abs(Unext[i]-U[i]);
                if(err>maxerr)maxerr=err;
            }
            swap(U,Unext);
        }while(maxerr>EPS);
        t+=tau;
#ifdef WRITE_TO_FILE
        if(counter%writeeach==0)
        {
            for(int i=0;i<Nn;i++)
            ofile<<U[i]<<endl;
            ofile<<endl;
            ofile<<endl;
        }
        counter++;
#endif
    }
    cputime=(double)(clock()-cputime)/CLOCKS_PER_SEC;
#ifdef WRITE_TO_FILE
    ofile.close();
#endif

    delete[] U;
    delete[] Unext;
    delete[] M;
    delete[] F;
    return cputime;
}

int main()
{
    float gpu,gpushared,cpu;
    gpu=solveGPU(1.0,50.0,0.01,1000000);
    cout<<"GPU Time: "<<gpu<<endl;
    gpushared=solveGPUshared(1.0,50.0,0.01,1000000);
    cout<<"GPU Time: "<<gpushared<<endl;
    cpu=solveCPU(1.0,50.0,0.01,1000000);
    cout<<"CPU Time: "<<cpu<<endl;
    cout<<"Max ratio:"<<cpu/min(gpu,gpushared)<<endl;
    return 0;
}
