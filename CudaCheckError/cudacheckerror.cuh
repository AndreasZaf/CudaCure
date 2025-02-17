#ifndef CUDAERROR_H
#define CUDAERROR_H

#include <stdio.h>          
#include <cuda.h>           
#include <cuda_runtime.h>   

// Function prototype for error checking
void CudaErrors(cudaError_t cudaStat);

#endif // CUDAERROR_H
