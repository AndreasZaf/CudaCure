#include "cudacheckerror.cuh" 

// Function to check CUDA errors
void CudaErrors(cudaError_t cudaStat)
{  
    // Check if the CUDA call was successful
    if (cudaStat != cudaSuccess)
    {
        // If there was an error, print the error message to the console
        printf("%s\n", cudaGetErrorString(cudaStat));
        
        exit(1);
    }
} 