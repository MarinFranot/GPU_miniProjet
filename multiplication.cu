
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>
#include <fstream>


bool checkCuda(int* out_cpu, int* out_gpu, int N);

#define CHK(code) \
do { \
    if ((code) != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s %s %i\n", \
                        cudaGetErrorString((code)), __FILE__, __LINE__); \
        goto Error; \
    } \
} while (0)



__global__ void addKernel(int* c, const int* a, const int* b, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;

    c[i] = a[i] + b[i];
}

__global__ void addKernel2(int* c, const int* a, const int* b, int N, int J)
{
    int i = (blockDim.x * blockIdx.x + threadIdx.x) * J;
    if (i >= N)
        return;
    for (int j = 0; j < J; j++)
        c[i + j] = a[i + j] + b[i + j];
}

__global__ void addKernel3(int* c, const int* a, const int* b, int N, int K)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;

    c[i] = a[i];
    for (int k = 0; k < K; k++)
        c[i] += b[i];
}

__global__ void mulKernel(int* c, const int* a, const int* b, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;

    c[i] = a[i] * b[i];
}

__global__ void mulKernel2(int* c, const int* a, const int* b, int N, int J)
{
    int i = (blockDim.x * blockIdx.x + threadIdx.x) * J;
    if (i >= N)
        return;
    for (int j = 0; j < J; j++)
        c[i + j] = a[i + j] * b[i + j];
}

__global__ void mulKernel3(int* c, const int* a, const int* b, int N, int K)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;

    c[i] = a[i];
    for (int k = 0; k < K; k++)
        c[i] *= b[i];
}

#define N_THREAD 128
#define Ntest 10


int main()
{
    int MODE = 3; // MODE: {1: simple addition, 2:addition avec du J, 3: addition avec du K}
                  // Si MODE = 1, alors les calcules font aussi être sur le CPU
                  // Si MODE > 1, le CPU ne va pas être utilisé
    int OP = 2;   // Op: Opérator: {1: addition, 2:multiplication}
    int J = 1;
    int K = 200;

    std::string name = "result/K_mul.txt";
    std::ofstream fichier(name, std::ios::out | std::ios::trunc);
    // fichier << "taille|temps_cpu|temps_gpu|memoryThroughput(GB/s)|computationThroughput(GOPS/s)" << std::endl;
    fichier << "taille|K|temps_gpu|memoryThroughput(GB/s)|computationThroughput(GOPS/s)|computeIntensity(OPS/Byte)" << std::endl;
    fichier << "Operator = " << OP << " | ";
    fichier << "Ntest = " << Ntest << " | ";
    fichier << "Mode = " << MODE << " | ";
    fichier << "N_thread = " << N_THREAD << std::endl;

    for (int puissance = 6; puissance < 7; puissance++) {
    
        // std::cout << puissance << std::endl;
        
        for (int k = 1; k < K; k++) {

            const int arraySize = pow(10, puissance); // On fait une taille de 10^(puissance)
            std::cout << k << std::endl;

            float temps_cpu = 0;
            float temps_gpu = 0;

            for (int nbtest = 0; nbtest < Ntest; nbtest++) {

                int* h_a = (int*)malloc(arraySize * sizeof(int));
                int* h_b = (int*)malloc(arraySize * sizeof(int));
                int* h_c = (int*)malloc(arraySize * sizeof(int));


                for (int i = 0; i < arraySize; i++) {
                    h_a[i] = 1 + i % 100;
                    h_b[i] = 1 + (arraySize - i) % 100;
                }
                
                //Computation on CPU
                if (MODE == 1) {

                    int* h_cpu_result = (int*)malloc(arraySize * sizeof(int));

                    std::chrono::steady_clock::time_point start_cpu = std::chrono::high_resolution_clock::now();

                    for (int i = 0; i < arraySize; i++) {
                        h_cpu_result[i] = h_a[i] * h_b[i];
                    }
                    std::chrono::steady_clock::time_point stop_cpu = std::chrono::high_resolution_clock::now();
                    auto cpu_runtime_us = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu).count();

                    temps_cpu += cpu_runtime_us;
                }


                //2. Do the computation on GPU and time it

                // Define the variable we need 
                int* dev_a = 0;
                int* dev_b = 0;
                int* dev_c = 0;

                cudaError_t cudaStatus;
                cudaEvent_t start_gpu, stop_gpu; //cudaEvent are used to time the kernel
                cudaEventCreate(&start_gpu);
                cudaEventCreate(&stop_gpu);

                // Define the size of the grid (block_size = #blocks in the grid)
                //and the size of a block (thread_size = #threads in a block)
                //TODO 1) Change how block_size and thread_size are defined to work with bigger vectors 
                dim3 block_size((arraySize + (N_THREAD - 1)) / N_THREAD);
                dim3 thread_size(N_THREAD);

                // Choose which GPU to run on, change this on a multi-GPU system.
                CHK(cudaSetDevice(0));


                // Allocate GPU buffers for three vectors (two input, one output)    .
                CHK(cudaMalloc((void**)&dev_c, arraySize * sizeof(int)));
                CHK(cudaMalloc((void**)&dev_a, arraySize * sizeof(int)));
                CHK(cudaMalloc((void**)&dev_b, arraySize * sizeof(int)));

                // Copy input vectors from host memory to GPU buffers.
                CHK(cudaMemcpy(dev_a, h_a, arraySize * sizeof(int), cudaMemcpyHostToDevice));
                CHK(cudaMemcpy(dev_b, h_b, arraySize * sizeof(int), cudaMemcpyHostToDevice));;

                // Launch a kernel on the GPU with one thread for each element and time the kernel
                cudaEventRecord(start_gpu);

                if (OP == 1) {  // Addition
                    if (MODE == 1)
                        addKernel << <block_size, thread_size >> > (dev_c, dev_a, dev_b, arraySize);
                    if (MODE == 2)
                        addKernel2 << <block_size, thread_size >> > (dev_c, dev_a, dev_b, arraySize, J);
                    if (MODE == 3)
                        addKernel3 << <block_size, thread_size >> > (dev_c, dev_a, dev_b, arraySize, K);
                }
                if (OP == 2) {  // Multiplication
                    if (MODE == 1)
                        mulKernel << <block_size, thread_size >> > (dev_c, dev_a, dev_b, arraySize);
                    if (MODE == 2)
                        mulKernel2 << <block_size, thread_size >> > (dev_c, dev_a, dev_b, arraySize, J);
                    if (MODE == 3)
                        mulKernel3 << <block_size, thread_size >> > (dev_c, dev_a, dev_b, arraySize, k);
                }

                cudaEventRecord(stop_gpu);


                // Check for any errors launching the kernel
                CHK(cudaGetLastError());


                // cudaDeviceSynchronize waits for the kernel to finish, and returns
                // any errors encountered during the launch.
                CHK(cudaDeviceSynchronize());


                // Copy output vector from GPU buffer to host memory.
                CHK(cudaMemcpy(h_c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost));


                // Make sure the stop_gpu event is recorded before doing the time computation
                cudaEventSynchronize(stop_gpu);
                float gpu_runtime_ms;
                cudaEventElapsedTime(&gpu_runtime_ms, start_gpu, stop_gpu);


                temps_gpu += gpu_runtime_ms;

                /*
                if (!checkCuda(h_cpu_result, h_c, arraySize)) {
                    printf("ERROR GPU results are not corrrrrect !!!\n");
                }
                */
                

            Error:
                cudaFree(dev_c);
                cudaFree(dev_a);
                cudaFree(dev_b);

                delete[] h_a, h_b, h_c;

                // cudaDeviceReset must be called before exiting in order for profiling and
                // tracing tools such as Nsight and Visual Profiler to show complete traces.
                cudaStatus = cudaDeviceReset();
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaDeviceReset failed!");
                    return 1;
                }
            }

            temps_cpu /= Ntest;
            temps_gpu *= 1000 / Ntest;

            float memoryUsed = 3.0 * arraySize * sizeof(int);
            float memoryThroughput = memoryUsed / temps_gpu / 1e+3; //Divide by 1 000 000 to have GB/s

            float numOperation = 1.0 * arraySize * k;
            // float memoryThroughput = memoryUsed / gpu_runtime_ms / 1e+6; //Divide by 1 000 000 to have GB/s
            float computationThroughput = numOperation / temps_gpu / 1e+3; // diviser par 1 000 car temps_gpu est en us
            float computeIntensity = computationThroughput / memoryThroughput;
            
            
            // std::cout << "Memory throughput : " << memoryThroughput << " GB/s " << std::endl;
            std::cout << "Computation throughput : " << computationThroughput << " GOPS/s " << std::endl;
            std::cout << "Compute intensity : " << computeIntensity << " OPS/Byte" << std::endl;

            /*
            std::cout << "mean: " << std::endl;
            std::cout << "Memory throughput : " << memoryThroughput << " GB/s " << std::endl;
            std::cout << "Computation throughput : " << computationThroughput << " GOPS/s " << std::endl;
            */

            // sauvegarde
            fichier << arraySize << "|";
            fichier << k << "|";
            fichier << temps_gpu << "|";
            fichier << memoryThroughput << "|";
            fichier << computationThroughput << "|";
            fichier << computeIntensity << std::endl;
        }
    }

    return 0;
}

bool checkCuda(int* out_cpu, int* out_gpu, int N) {
    bool res = true;
    for (int i = 0; i < N; i++) {
        if (out_cpu[i] != out_gpu[i]) {
            printf("ERROR : cpu : %d != gpu : %d \n", out_cpu[i], out_gpu[i]);
            res = false;
        }
    }
    return res;
}
