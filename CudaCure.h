/*------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------CudaCurePart-------------------------------------------------------------------------------------------------*/
#ifndef GPUINITIALIZATIONPHASE_H
#define GPUINITIALIZATIONPHASE_H
void GPU_initialization_phase(struct BTree *item,struct nnb_info *GPUnnb,int npat,int rsize,int norm,int lpat,double *kernel_timer);
#endif

#ifndef GPUMINIMUMDISTANCEPAIR
#define GPUMINIMUMDISTANCEPAIR_H
void GPU_minimum_distance_pair(struct BTree *item,struct nnb_info *GPUnnb,int npat,int *gpair1,double *gmin_dist,double *kernel_timer);
#endif

#ifndef GPUUPDATENNB
#define GPUUPDATENNB
void GPU_update_nnb(struct BTree *item,struct nnb_info *GPUnnb,int npat,int rsize,int norm,int lpat,int pair1,int pair2,int *gpair1,double *gmin_dist,double *kernel_timer);
#endif 

#ifndef GPUFIRSTPRUNING
#define GPUFIRSTPRUNING
void GPU_first_pruning(struct BTree *item,int npat,int *nodes,int *pruned_nodes,double *kernel_timer);
#endif 

#ifndef GPUSECONDPRUNING
#define GPUSECONDPRUNING
void GPU_second_pruning(struct BTree *item,int npat,int *nodes,int *pruned_nodes,double *kernel_timer);
#endif 

#ifndef GPUPRUNING
#define GPUPRUNING
void GPU_pruning(struct BTree *item,struct nnb_info *GPUnnb,int npat,int rsize,int norm,int lpat,int *gpair1,double *gmin_dist,double *kernel_timer);
#endif 

#ifndef CUDACUREPARTRESULTS
#define CUDACUREPARTRESULTS

void Gpu_Cuda_Cure_Part_Results(int npat,int lpat,int csize,double init_timer,double find_mdp_timer,double clustering_timer,double merge_timer,double update_timer,double pruning_timer,const char *type);

#endif

#ifndef CUDACUREPRINTRESULTS
#define CUDACUREPRINTRESULTS

void Gpu_Cuda_Cure_Print_Results(struct BTree *item,int npat,int rsize,int lpat,int clusters,const char *type);

#endif

/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------CudaCureAllDataGpu---------------------------------------------------------------------------------------------------*/

#ifndef ALLDATAGPU_H
#define ALLDATAGPU_H

void GPU_Clustering(struct BTree *item,int npat,int rsize,int norm,int lpat,int csize);

#endif


/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/


/*---------------------------------------------------------CudaCureUnifiedMemory-----------------------------------------------------------------------------------------------*/
#ifndef UNIFIEDMEMORY_H
#define UNIFIEDMEMORY_H

void GPU_Clustering_Unified_Memory(struct BTree *item,int npat,int rsize,int norm,int lpat,int csize);

#endif

/*-------------------------------------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------PrefetchAsyncCudaCureAllDataGpuUnifiedMemory----------------------------------------------------------------------------*/
#ifndef PREFETCHASYNCUNIFIEDMEMORY_H
#define PREFETCHASYNCUNIFIEDMEMORY_H

void Prefetch_Async_GPU_Clustering_Unified_Memory(struct BTree *item,int npat,int rsize,int norm,int lpat,int csize);

#endif

/*--------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------CudaCurePinnedMemoryCMH---------------------------------------------------------------*/

#ifndef PINNEDMEMORYCMH_H
#define PINNEDMEMORYCMH_H

void GPU_Clustering_Pinned_Memory_CMH(struct BTree *item,int npat,int rsize,int norm,int lpat,int csize);

#endif

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/

/*----------------------------------------------------------------CudaCurePinnedMemoryCHA----------------------------------------------------------*/

#ifndef PINNEDMEMORYCHA_H
#define PINNEDMEMORYCHA_H
void GPU_Clustering_Pinned_Memory_CHA(struct BTree *item,int npat,int rsize,int norm,int lpat,int csize);
#endif
/*------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/*----------------------------------------------------------------------SingleCudaCure------------------------------------------------------------------------------------*/
#ifndef SINGLECUDACURE_H
#define SINGLECUDACURE_H
void Gpu_Single_Clustering(struct BTree *item,int npat,int rsize,int norm,int lpat,int csize,const char *type);
#endif
/*----------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/*----------------------------------------------------------------CudaCureMappedPinnedMemory------------------------------------------------------------------*/

#ifndef MAPPEDPINNEDMEMORY1_H
#define MAPPEDPINNEDMEMORY1_H

void GPU_Clustering_Mapped_Pinned_Memory(struct BTree *item,int npat,int rsize,int norm,int lpat,int csize);

#endif
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/*--------------------------------------------------------------------GpuId--------------------------------------------------------------------------------*/
#ifndef ID_DEVICE
#define ID_DEVICE
void Id_Device(int *id,int number_of_gpus);
#endif
/*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/*--------------------------------------------------------------------NumberOfGpus--------------------------------------------------------------------------------*/
#ifndef NUMBEROFGPUS
#define NUMBEROFGPUS
int Number_Of_Gpus();
#endif
/*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------*/