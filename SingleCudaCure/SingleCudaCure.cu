#include "../CudaCheckError/cudacheckerror.cuh" 
#include "../CudaTimer/cudatimer.cuh"  
#include "../CudaKernelsHostDeviceFunctions/KernelsHostDeviceFunctions.cuh" 

extern"C"
{
#include "../CudaCure.h"  // Header for core CURE algorithm functions

}

extern "C"
{

// Main GPU-based clustering function that routes to specific clustering implementation
void Gpu_Single_Clustering(struct BTree *item,int npat,int rsize,int norm,int lpat,int csize,const char *type)
{

	 if(strcmp(type,"ALLCUDACURE")==0)
	 {
	     // Call the regular GPU-based CURE clustering method
		GPU_Clustering(item,npat,rsize,norm,lpat,csize);
		
	 }else if(strcmp(type,"UMCUDACURE")==0)
	 {
	   // Call the GPU CURE clustering method using unified memory
	   GPU_Clustering_Unified_Memory(item,npat,rsize,norm,lpat,csize);
	 
	 }else if(strcmp(type,"PAUMCUDACURE")==0)
	 {
	    // Call the GPU CURE clustering method with prefetch async and unified memory
		Prefetch_Async_GPU_Clustering_Unified_Memory(item,npat,rsize,norm,lpat,csize);
	   
	 }else if(strcmp(type,"CMHCUDACURE")==0)
	 {
	    // Call the GPU CURE clustering method using pinned memory
		GPU_Clustering_Pinned_Memory_CMH(item,npat,rsize,norm,lpat,csize);
		
	 }else if(strcmp(type,"CHACUDACURE")==0)
	 {
	    // Call the GPU CURE clustering method using pinned memory
		GPU_Clustering_Pinned_Memory_CHA(item,npat,rsize,norm,lpat,csize);
		
	 }else if(strcmp(type,"MCUDACURE")==0)
	 {
	    // Call the GPU CURE clustering method using mapped pinned memory
		GPU_Clustering_Mapped_Pinned_Memory(item,npat,rsize,norm,lpat,csize);
			
	 }	
}

}