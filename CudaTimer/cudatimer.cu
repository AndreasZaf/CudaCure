#include "cudatimer.cuh"

// Function to create and initialize the CUDA timer events
void CreateTimer(CudaTimer* Timer)
{
    // Create a CUDA event for the start timer
    CudaErrors(cudaEventCreate(&Timer->Starttimer));
    
    // Create a CUDA event for the end timer
    CudaErrors(cudaEventCreate(&Timer->Endtimer));
}

// Function to destroy the CUDA timer events and free resources
void DestroyTimer(CudaTimer* Timer)
{
    // Destroy the start timer event
    CudaErrors(cudaEventDestroy(Timer->Starttimer));
    
    // Destroy the end timer event
    CudaErrors(cudaEventDestroy(Timer->Endtimer));
}

// Function to start the timer using the specified CUDA stream
void StartTimer(CudaTimer* Timer, cudaStream_t stream)
{
    // If no stream is specified (stream == 0), record the start event without a stream
    if (stream == 0)
    {
        CudaErrors(cudaEventRecord(Timer->Starttimer));
    }
    else
    {
        // Record the start event on the specified stream
        CudaErrors(cudaEventRecord(Timer->Starttimer, stream));
    }
}

// Function to stop the timer using the specified CUDA stream
void StopTimer(CudaTimer* Timer, cudaStream_t stream)
{
    // If no stream is specified (stream == 0), record the end event without a stream
    if (stream == 0)
    {
        CudaErrors(cudaEventRecord(Timer->Endtimer));
    }
    else
    {
        // Record the end event on the specified stream
        CudaErrors(cudaEventRecord(Timer->Endtimer, stream));
    }
    
    // Wait for the end event to complete before proceeding
    CudaErrors(cudaEventSynchronize(Timer->Endtimer));
}

// Function to retrieve the elapsed time in seconds between the start and stop events
double GetElapsedTime(CudaTimer* Timer)
{
    float elapsedTime;

    // Calculate the elapsed time in milliseconds between start and end events
    CudaErrors(cudaEventElapsedTime(&elapsedTime, Timer->Starttimer, Timer->Endtimer));
    
    // Convert milliseconds to seconds and return the result
    return (double)elapsedTime / 1000.0;
}