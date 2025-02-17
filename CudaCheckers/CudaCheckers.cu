#include "../CudaCheckError/cudacheckerror.cuh" 
#include "../CudaKernelsHostDeviceFunctions/KernelsHostDeviceFunctions.cuh" 

extern "C" 
{
    #include "../CudaCure.h" // Include the CudaCure header file
}

extern "C" 
{

    // Function to get the number of available CUDA-capable GPUs
    int Number_Of_Gpus()
    {
        int deviceCount; // Variable to store the number of devices

        
        CudaErrors(cudaGetDeviceCount(&deviceCount));
        
        // Return the count of CUDA devices
        return deviceCount;
    }
  
    // Function to get the device IDs of all available CUDA devices
    void Id_Device(int *id, int number_of_gpus)
    {
      
        for (unsigned int i = 0; i < number_of_gpus; i++)
        {
            int deviceID; // Variable to store the current device ID
            
            
            CudaErrors(cudaGetDevice(&deviceID));
            
            // Store the device ID in the provided array
            id[i] = deviceID;
        }
    }
    
}
