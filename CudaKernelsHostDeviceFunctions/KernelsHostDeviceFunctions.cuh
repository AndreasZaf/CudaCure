#ifndef CUDAKERNELSHOSTDEVICEFUNCTIONS_H

#define CUDAKERNELSHOSTDEVICEFUNCTIONS_H

#include <unistd.h>

#include "../cluster.h"
#include "../help.h"

#include "../CudaCheckError/cudacheckerror.cuh"
#include "../Timers/Timers.h"

__host__ DATATYPE one_minus_alpha_rev_value();
__host__ char *current_path();
__host__ void transfer_data_to_constant_memory(DATATYPE alpha,DATATYPE one_minus_alpha_rev);
__host__ void transfer_data_to_constant_memory(int lpat,int rsize,int norm);
__host__ double* Timer_Array(int timers);
__host__ void TimersFree(struct Timers *BT);
__host__ BTree* alloc_item(struct BTree *item,int npat,int rsize,int lpat);
__host__ void FreeItems(struct BTree *item,int npat,int rsize);
__host__ void change_directory(const char *path);
__host__ char* ConcenateString(const char *str1,const char *str2);

__host__ void show_time_results_for_each_case(const char *type,struct Timers *BT,int npat,int lpat,int csize);

__host__ void print_rep_points(FILE *f, struct BTree *item, int cno, int rsize,int lpat);
__host__ void print_points(FILE *f, struct BTree *item, int cno, int lpat);
__host__ void print_results(const char *type, struct BTree *item,int npat,int clusters,int rsize,int lpat);

__device__ double GPUdistance(DATATYPE *pat1, DATATYPE *pat2);
__device__ double GPU_cure_distance(struct BTree *x,struct BTree *y);

__global__ void GPUroot(double *dist,double *gmin_dist);
__global__ void GPUmerge(struct BTree *item,struct BTree *merge_tmp_item,int **pats,int pair1,int pair2,double *min_dist,int size_limit);
__global__ void BTree_transfer(struct BTree *item,DATATYPE *rep,DATATYPE *mean,int rsize,int lpat);
__global__ void pats_transfer(int **pats,int *pats1);
__global__ void GPU_initialization_phase1(struct BTree *item,struct nnb_info *GPU_nnb,int npat);
__global__ void NO_NONE_MDP(struct BTree *item,struct nnb_info *GPU_nnb,int *GPU_gpair1,double *GPU_gmin_dist,int *counter,int npat);
__global__ void GPU_find_minimum_distance_pair_per_block(int *GPU_lpair1,double *GPU_lmin_dist,int *GPU_gpair1,double *GPU_gmin_dist,int npat);
__global__ void GPU_find_minimum_distance_pair(int *index ,double *distances,int npat);
__global__ void GPU_update_nnb_kernel(struct BTree *item,struct nnb_info *GPU_nnb,int npat,int *GPU_gpair1,double *GPU_gmin_dist);
__global__ void GPU_update_nnb_kernel1(struct BTree *item,struct nnb_info *GPU_nnb,int npat,int pair1,int pair2,int *GPU_gpair1,double *GPU_gmin_dist,int offset);
__global__ void GPU_update_nnb_kernel2(struct BTree *item,struct nnb_info *GPU_nnb,int npat,int pair1,int pair2,int *GPU_gpair1,double *GPU_gmin_dist,int offset);
__global__ void GPU_update_nnb_kernel3(struct BTree *item,struct nnb_info *GPU_nnb,int npat,int pair1,int pair2,int offset,int *counter,int*indices);
__global__ void GPU_update_kernel(struct BTree *item,int npat,int *indices,double *distances,int *counter);
__global__ void GPU_update_kernel1(int npat,int *indices,double *distances);
__global__ void GPU_update_kernel2(struct BTree *item,struct nnb_info *GPU_nnb,int node,int *indices,double *distances,int *lpair1,double *lmin_dist);
__global__ void GPU_update_kernel3(int *lpair1,double *lmin_dist,int *indices,double *distances,int npat,int *counter);
__global__ void GPU_update_kernel4(int *indices,double *distances,int npat,int counter);
__global__ void GPU_first_pruning_kernel(struct BTree *item,int npat,int*nodes,int *pruned_nodes);
__global__ void GPU_second_pruning_kernel(struct BTree *item,int npat,int *nodes,int *pruned_nodes);
__global__ void GPU_pruning_kernel1(struct BTree *item,struct nnb_info *GPU_nnb,int npat,int *counter,int *indices);
__global__ void GPU_pruning_kernel2(struct BTree *item,struct nnb_info *GPUnnb,int *indices,int npat,int *lpair1,double *lmin_dist1);
#endif