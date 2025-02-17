#ifndef CUDATIMER_H
#define CUDATIMER_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "../CudaCheckError/cudacheckerror.cuh"

// Structure to hold CUDA timer information
struct CudaTimer
{
    // CUDA event for starting the timer
    cudaEvent_t Starttimer;
    
    // CUDA event for stopping the timer
    cudaEvent_t Endtimer;
};

// Function prototypes for timer management
// Creates and initializes a CudaTimer
void CreateTimer(CudaTimer* Timer);

// Destroys a CudaTimer and frees resources
void DestroyTimer(CudaTimer* Timer);

// Starts the timer using a specific CUDA stream
void StartTimer(CudaTimer* Timer, cudaStream_t stream);

// Stops the timer using a specific CUDA stream
void StopTimer(CudaTimer* Timer, cudaStream_t stream);

// Returns the elapsed time in seconds between the start and stop events
double GetElapsedTime(CudaTimer* Timer);

#endif // CUDATIMER_H
