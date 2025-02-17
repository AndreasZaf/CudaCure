#include "../CudaCheckError/cudacheckerror.cuh"
#include "../CudaTimer/cudatimer.cuh"
#include "../CudaKernelsHostDeviceFunctions/KernelsHostDeviceFunctions.cuh"

extern "C"{

#include "../CudaCure.h"

}

extern "C"
{
void GPU_initialization_phase(struct BTree *item,struct nnb_info *GPUnnb,int npat,int rsize,int norm,int lpat,double *kernel_timer)
{

struct BTree *GPUitem;
struct nnb_info *KERNEL_GPUnnb;

CudaErrors(cudaMalloc((void**)&GPUitem,npat*sizeof(BTree)));
CudaErrors(cudaMemcpy(GPUitem,item,npat*sizeof(BTree),cudaMemcpyHostToDevice));

DATATYPE **rep;
DATATYPE *rep1;

for(unsigned int i=0;i<npat;i++)
{
CudaErrors(cudaMalloc((void**)&rep,item[i].size*sizeof(DATATYPE*)));

for(unsigned int j=0;j<item[i].size;j++)
{  
CudaErrors(cudaMalloc((void**)&rep1,lpat*sizeof(DATATYPE)));   
CudaErrors(cudaMemcpy(rep1,item[i].rep[j],lpat*sizeof(DATATYPE),cudaMemcpyHostToDevice)); 
CudaErrors(cudaMemcpy(&rep[j],&rep1,sizeof(DATATYPE*),cudaMemcpyHostToDevice)); 
}

CudaErrors(cudaMemcpy(&(GPUitem[i].rep),&rep,sizeof(DATATYPE**),cudaMemcpyHostToDevice));
}

transfer_data_to_constant_memory(lpat,rsize,norm);

CudaErrors(cudaMalloc((void**)&KERNEL_GPUnnb,npat*sizeof(nnb_info)));

gridsize=(npat+blocksize-1)/blocksize;

CudaTimer timer;

CreateTimer(&timer);
StartTimer(&timer,0);   

GPU_initialization_phase1<<<gridsize,blocksize>>>(GPUitem,KERNEL_GPUnnb,npat);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
*kernel_timer=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaMemcpy(GPUnnb,KERNEL_GPUnnb,npat*sizeof(nnb_info),cudaMemcpyDeviceToHost));

CudaErrors(cudaFree(rep1));
CudaErrors(cudaFree(rep));
CudaErrors(cudaFree(GPUitem));

CudaErrors(cudaFree(KERNEL_GPUnnb));
}

void GPU_minimum_distance_pair(struct BTree *item,struct nnb_info *GPUnnb,int npat,int *gpair1,double *gmin_dist,double *kernel_timer)
{
struct BTree *GPUitem;
struct nnb_info *KERNEL_GPUnnb;

int *block_index_minimum;
int *lpair1;
int *GPU_counter;

int counter=0;

double *block_distance_minimum;
double *lmin_dist;

CudaTimer timer;

CudaErrors(cudaMalloc(&GPUitem,npat*sizeof(BTree)));
CudaErrors(cudaMemcpy(GPUitem,item,npat*sizeof(BTree),cudaMemcpyHostToDevice));

CudaErrors(cudaMalloc((void**)&KERNEL_GPUnnb,npat*sizeof(nnb_info)));
CudaErrors(cudaMemcpy(KERNEL_GPUnnb,GPUnnb,npat*sizeof(nnb_info),cudaMemcpyHostToDevice));

CudaErrors(cudaMalloc((void**)&lpair1,npat*sizeof(int)));
CudaErrors(cudaMalloc((void**)&lmin_dist,npat*sizeof(double)));

CudaErrors(cudaMalloc((void**)&GPU_counter,sizeof(int)));

gridsize=(npat+blocksize-1)/blocksize;

CudaErrors(cudaMalloc((void**)&block_index_minimum,gridsize*sizeof(int)));
CudaErrors(cudaMalloc((void**)&block_distance_minimum,gridsize*sizeof(double)));

CreateTimer(&timer);
StartTimer(&timer,0);

NO_NONE_MDP<<<gridsize,blocksize>>>(GPUitem,KERNEL_GPUnnb,lpair1,lmin_dist,GPU_counter,npat);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaMemcpy(&counter,GPU_counter,sizeof(int),cudaMemcpyDeviceToHost));

gridsize=(counter+blocksize-1)/blocksize;

CreateTimer(&timer);

StartTimer(&timer,0);   

GPU_find_minimum_distance_pair_per_block<<<gridsize,blocksize>>>(lpair1,lmin_dist,block_index_minimum,block_distance_minimum,counter);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaFree(lpair1));
CudaErrors(cudaFree(lmin_dist));

if(gridsize>1)
{
CudaErrors(cudaFree(GPUitem));
CudaErrors(cudaFree(KERNEL_GPUnnb));

while(gridsize>1)
{
new_gridsize=(gridsize+blocksize-1)/blocksize;

CreateTimer(&timer);
StartTimer(&timer,0); 

GPU_find_minimum_distance_pair<<<new_gridsize,blocksize>>>(block_index_minimum,block_distance_minimum,gridsize);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);

*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

gridsize=new_gridsize;
}
   
}else
{
CudaErrors(cudaFree(GPUitem));
CudaErrors(cudaFree(KERNEL_GPUnnb));
}

CudaErrors(cudaMemcpy(gpair1,block_index_minimum,sizeof(int),cudaMemcpyDeviceToHost));
CudaErrors(cudaMemcpy(gmin_dist,block_distance_minimum,sizeof(double),cudaMemcpyDeviceToHost));

CudaErrors(cudaFree(block_index_minimum));
CudaErrors(cudaFree(block_distance_minimum));
}

void GPU_update_nnb(struct BTree *item,struct nnb_info *GPUnnb,int npat,int rsize,int norm,int lpat,int pair1,int pair2,int *gpair1,double *gmin_dist,double *kernel_timer)
{
struct BTree *GPUitem;
struct nnb_info *KERNEL_GPUnnb;

int *block_index_minimum;
int *lpair1;
int *GPU_counter;
int *indices;
int *temp_indices;

int counter=0;
int counter1=0;
int node;

double *block_distance_minimum;
double *lmin_dist;
double *temp_distances;

cudaStream_t streams[3];

for(unsigned int i=0; i<3; i++)
{
CudaErrors(cudaStreamCreate(&streams[i]));
}

CudaErrors(cudaMalloc((void**)&lpair1,npat*sizeof(int)));
CudaErrors(cudaMalloc((void**)&indices,npat*sizeof(int)));
CudaErrors(cudaMalloc((void**)&lmin_dist,npat*sizeof(double)));  

CudaErrors(cudaMalloc((void**)&GPUitem,npat*sizeof(BTree)));

CudaErrors(cudaMemcpyAsync(GPUitem,item,pair2*sizeof(BTree),cudaMemcpyHostToDevice,streams[0]));

DATATYPE **rep;
DATATYPE *rep1;

for(unsigned int i=0;i<pair2;i++)
{

unsigned int maxrep=(item[i].size<=rsize)?item[i].size:rsize;

CudaErrors(cudaMalloc((void**)&rep,maxrep*sizeof(DATATYPE*)));

for(unsigned int j=0;j<maxrep;j++)
{

CudaErrors(cudaMalloc((void**)&rep1,lpat*sizeof(DATATYPE)));

CudaErrors(cudaMemcpyAsync(rep1,item[i].rep[j],lpat*sizeof(DATATYPE),cudaMemcpyHostToDevice,streams[0])); 
CudaErrors(cudaMemcpyAsync(&rep[j],&rep1,sizeof(DATATYPE*),cudaMemcpyHostToDevice,streams[0])); 
}

CudaErrors(cudaMemcpyAsync(&(GPUitem[i].rep),&rep,sizeof(DATATYPE**),cudaMemcpyHostToDevice,streams[0]));

} 

CudaErrors(cudaMemcpyAsync(GPUitem+pair2,item+pair2,(pair1-pair2)*sizeof(BTree),cudaMemcpyHostToDevice,streams[1]));

for(unsigned int i=pair2;i<pair1;i++)
{


unsigned int maxrep=(item[i].size<=rsize)?item[i].size:rsize;

CudaErrors(cudaMalloc((void**)&rep,maxrep*sizeof(DATATYPE*)));

for(unsigned int j=0;j<maxrep;j++)
{

CudaErrors(cudaMalloc((void**)&rep1,lpat*sizeof(DATATYPE)));

CudaErrors(cudaMemcpyAsync(rep1,item[i].rep[j],lpat*sizeof(DATATYPE),cudaMemcpyHostToDevice,streams[1])); 
CudaErrors(cudaMemcpyAsync(&rep[j],&rep1,sizeof(DATATYPE*),cudaMemcpyHostToDevice,streams[1]));  
}

CudaErrors(cudaMemcpyAsync(&(GPUitem[i].rep),&rep,sizeof(DATATYPE**),cudaMemcpyHostToDevice,streams[1]));

} 

CudaErrors(cudaMemcpyAsync(GPUitem+pair1,item+pair1,(npat-pair1)*sizeof(BTree),cudaMemcpyHostToDevice,streams[2]));

for(unsigned int i=pair1;i<npat;i++)
{

unsigned int maxrep=(item[i].size<=rsize)?item[i].size:rsize;

CudaErrors(cudaMalloc((void**)&rep,maxrep*sizeof(DATATYPE*)));

for(unsigned int j=0;j<maxrep;j++)
{

CudaErrors(cudaMalloc((void**)&rep1,lpat*sizeof(DATATYPE)));

CudaErrors(cudaMemcpyAsync(rep1,item[i].rep[j],lpat*sizeof(DATATYPE),cudaMemcpyHostToDevice,streams[2])); 
CudaErrors(cudaMemcpyAsync(&rep[j],&rep1,sizeof(DATATYPE*),cudaMemcpyHostToDevice,streams[2]));
}

CudaErrors(cudaMemcpyAsync(&(GPUitem[i].rep),&rep,sizeof(DATATYPE**),cudaMemcpyHostToDevice,streams[2]));

} 

CudaErrors(cudaMalloc((void**)&KERNEL_GPUnnb,npat*sizeof(nnb_info)));

CudaErrors(cudaMemcpyAsync(KERNEL_GPUnnb,GPUnnb,pair2*sizeof(nnb_info),cudaMemcpyHostToDevice,streams[0]));
CudaErrors(cudaMemcpyAsync(KERNEL_GPUnnb+pair2,GPUnnb+pair2,(pair1-pair2)*sizeof(nnb_info),cudaMemcpyHostToDevice,streams[1]));
CudaErrors(cudaMemcpyAsync(KERNEL_GPUnnb+pair1,GPUnnb+pair1,(npat-pair1)*sizeof(nnb_info),cudaMemcpyHostToDevice,streams[2]));

CudaErrors(cudaMalloc((void**)&GPU_counter,sizeof(int)));
CudaErrors(cudaMemcpy(GPU_counter,&counter,sizeof(int),cudaMemcpyHostToDevice));

gridsize=(npat+blocksize-1)/blocksize;

CudaErrors(cudaMalloc((void**)&block_index_minimum,gridsize*sizeof(int)));
CudaErrors(cudaMalloc((void**)&block_distance_minimum,gridsize*sizeof(double)));

CudaErrors(cudaMalloc((void**)&temp_indices,gridsize*sizeof(int)));
CudaErrors(cudaMalloc((void**)&temp_distances,gridsize*sizeof(double)));

CudaTimer timer;

CreateTimer(&timer);
StartTimer(&timer,streams[0]);
GPU_update_nnb_kernel<<<(pair2+blocksize-1)/blocksize,blocksize,0,streams[0]>>>(GPUitem,KERNEL_GPUnnb,pair2,lpair1,lmin_dist);
CudaErrors(cudaStreamSynchronize(streams[0]));
StopTimer(&timer,streams[0]);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CreateTimer(&timer);
StartTimer(&timer,streams[1]);
GPU_update_nnb_kernel1<<<((pair1-pair2)+blocksize-1)/blocksize,blocksize,0,streams[1]>>>(GPUitem,KERNEL_GPUnnb,pair1,pair1,pair2,lpair1,lmin_dist,pair2);
CudaErrors(cudaStreamSynchronize(streams[1]));
StopTimer(&timer,streams[1]);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

transfer_data_to_constant_memory(lpat,rsize,norm);

CreateTimer(&timer);
StartTimer(&timer,streams[2]);
GPU_update_nnb_kernel2<<<((npat-pair1)+blocksize-1)/blocksize,blocksize,0,streams[2]>>>(GPUitem,KERNEL_GPUnnb,npat,pair1,pair2,lpair1,lmin_dist,pair1);
CudaErrors(cudaStreamSynchronize(streams[2]));
StopTimer(&timer,streams[2]);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaMemcpyAsync(GPUnnb+pair1,KERNEL_GPUnnb+pair1,(npat-pair1)*sizeof(nnb_info),cudaMemcpyDeviceToHost,streams[2]));

for(unsigned int i = 0; i<3; i++)
{
CudaErrors(cudaStreamDestroy(streams[i]));
}

CreateTimer(&timer);
StartTimer(&timer,0);
GPU_update_nnb_kernel3<<<((npat-pair2)+blocksize-1)/blocksize,blocksize>>>(GPUitem,KERNEL_GPUnnb,npat,pair1,pair2,pair2,GPU_counter,indices);
CudaErrors(cudaDeviceSynchronize());
StopTimer(&timer,0);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaMemcpy(&counter,GPU_counter,sizeof(int),cudaMemcpyDeviceToHost));

for(unsigned int i=0;i<counter;i++)
{

CudaErrors(cudaMemcpy(&node,&indices[i],sizeof(int),cudaMemcpyDeviceToHost));

counter1=0;
CudaErrors(cudaMemcpy(GPU_counter,&counter1,sizeof(int),cudaMemcpyHostToDevice)); 

CreateTimer(&timer);
StartTimer(&timer,0);

GPU_update_kernel<<<(node+blocksize-1)/blocksize+1,blocksize>>>(GPUitem,node,temp_indices,temp_distances,GPU_counter);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaMemcpy(&counter1,GPU_counter,sizeof(int),cudaMemcpyDeviceToHost));

CreateTimer(&timer);
StartTimer(&timer,0);
GPU_update_kernel4<<<((((node+blocksize-1)/blocksize)+1)+blocksize-1)/blocksize+1,blocksize>>>(temp_indices,temp_distances,(node+blocksize-1)/blocksize+1,counter1);
StopTimer(&timer,0);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

if(((node+blocksize-1)/blocksize+1)>1)
{

if(counter1>0)
{
unsigned int gridsizes5=(node+blocksize-1)/blocksize+1;

if(gridsizes5>counter1)
{
gridsizes5=counter1;
}

while(gridsizes5>1)
{
new_gridsize=(gridsizes5+blocksize-1)/blocksize;

CreateTimer(&timer);
StartTimer(&timer,0);

GPU_update_kernel1<<<new_gridsize,blocksize>>>(gridsizes5,temp_indices,temp_distances);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

gridsizes5=new_gridsize;
}

} 
}

CreateTimer(&timer);
StartTimer(&timer,0);

GPU_update_kernel2<<<1,1>>>(GPUitem,KERNEL_GPUnnb,node,temp_indices,temp_distances,lpair1,lmin_dist);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

}

CudaErrors(cudaFree(indices));

CudaErrors(cudaMemcpy(GPUnnb,KERNEL_GPUnnb,npat*sizeof(nnb_info),cudaMemcpyDeviceToHost));

counter=0;
CudaErrors(cudaMemcpy(GPU_counter,&counter,sizeof(int),cudaMemcpyHostToDevice));

CreateTimer(&timer);
StartTimer(&timer,0);  
GPU_update_kernel3<<<gridsize,blocksize>>>(lpair1,lmin_dist,block_index_minimum,block_distance_minimum,npat,GPU_counter);
CudaErrors(cudaDeviceSynchronize());
StopTimer(&timer,0);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaMemcpy(&counter,GPU_counter,sizeof(int),cudaMemcpyDeviceToHost));

unsigned int gridsizes5=(gridsize+blocksize-1)/blocksize+1;

CreateTimer(&timer);
StartTimer(&timer,0);
GPU_update_kernel4<<<gridsizes5,blocksize>>>(block_index_minimum,block_distance_minimum,gridsize,counter);
CudaErrors(cudaDeviceSynchronize());
StopTimer(&timer,0);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaFree(lpair1));
CudaErrors(cudaFree(lmin_dist));

CudaErrors(cudaFree(GPU_counter));

if(gridsize>1)
{

CudaErrors(cudaFree(rep1));
CudaErrors(cudaFree(rep));
CudaErrors(cudaFree(GPUitem));

CudaErrors(cudaFree(KERNEL_GPUnnb));

if(counter>0)
{

if(gridsize>counter)
{
gridsize=counter;
}

while(gridsize>1)
{
new_gridsize=(gridsize+blocksize-1)/blocksize;

CreateTimer(&timer);
StartTimer(&timer,0);

GPU_find_minimum_distance_pair<<<new_gridsize,blocksize>>>(block_index_minimum,block_distance_minimum,gridsize);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

gridsize=new_gridsize;
}

} 

}else

{
CudaErrors(cudaFree(rep1));
CudaErrors(cudaFree(rep));
CudaErrors(cudaFree(GPUitem));

CudaErrors(cudaFree(KERNEL_GPUnnb));
}

CudaErrors(cudaMemcpy(gpair1,block_index_minimum,sizeof(int),cudaMemcpyDeviceToHost));
CudaErrors(cudaMemcpy(gmin_dist,block_distance_minimum,sizeof(double),cudaMemcpyDeviceToHost));

CudaErrors(cudaFree(block_index_minimum));
CudaErrors(cudaFree(block_distance_minimum));

}

void GPU_first_pruning(struct BTree *item,int npat,int *nodes,int *pruned_nodes,double *kernel_timer)
{

struct BTree *GPUitem;

int *KERNEL_GPU_nodes;
int *KERNEL_GPU_pruned_nodes;

CudaErrors(cudaMalloc((void**)&KERNEL_GPU_nodes,sizeof(int)));
CudaErrors(cudaMemcpy(KERNEL_GPU_nodes,nodes,sizeof(int),cudaMemcpyHostToDevice));

CudaErrors(cudaMalloc((void**)&KERNEL_GPU_pruned_nodes,sizeof(int)));
CudaErrors(cudaMemcpy(KERNEL_GPU_pruned_nodes,pruned_nodes,sizeof(int),cudaMemcpyHostToDevice));

CudaErrors(cudaMalloc((void**)&GPUitem, npat * sizeof(BTree)));
CudaErrors(cudaMemcpy(GPUitem,item,npat*sizeof(BTree),cudaMemcpyHostToDevice));

gridsize=(npat+blocksize-1)/blocksize;

CudaTimer timer;

CreateTimer(&timer);
StartTimer(&timer,0);

GPU_first_pruning_kernel<<<gridsize,blocksize>>>(GPUitem,npat,KERNEL_GPU_nodes,KERNEL_GPU_pruned_nodes);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaMemcpy(item,GPUitem,npat*sizeof(BTree),cudaMemcpyDeviceToHost)); 
CudaErrors(cudaMemcpy(nodes,KERNEL_GPU_nodes,sizeof(int),cudaMemcpyDeviceToHost));
CudaErrors(cudaMemcpy(pruned_nodes,KERNEL_GPU_pruned_nodes,sizeof(int),cudaMemcpyDeviceToHost));

CudaErrors(cudaFree(GPUitem));
CudaErrors(cudaFree(KERNEL_GPU_nodes));
CudaErrors(cudaFree(KERNEL_GPU_pruned_nodes));

}

void GPU_second_pruning(struct BTree *item,int npat,int *nodes,int *pruned_nodes,double *kernel_timer)
{

struct BTree *GPUitem;

int *KERNEL_GPU_nodes;
int *KERNEL_GPU_pruned_nodes;

CudaErrors(cudaMalloc((void**)&KERNEL_GPU_nodes,sizeof(int)));
CudaErrors(cudaMemcpy(KERNEL_GPU_nodes,nodes,sizeof(int),cudaMemcpyHostToDevice));

CudaErrors(cudaMalloc((void**)&KERNEL_GPU_pruned_nodes,sizeof(int)));
CudaErrors(cudaMemcpy(KERNEL_GPU_pruned_nodes,pruned_nodes,sizeof(int),cudaMemcpyHostToDevice));

CudaErrors(cudaMalloc((void**)&GPUitem, npat * sizeof(BTree)));
CudaErrors(cudaMemcpy(GPUitem,item,npat*sizeof(BTree),cudaMemcpyHostToDevice));

gridsize=(npat+blocksize-1)/blocksize;

CudaTimer timer;

CreateTimer(&timer);
StartTimer(&timer,0);

GPU_second_pruning_kernel<<<gridsize,blocksize>>>(GPUitem,npat,KERNEL_GPU_nodes,KERNEL_GPU_pruned_nodes);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaMemcpy(item,GPUitem,npat*sizeof(BTree),cudaMemcpyDeviceToHost)); 
CudaErrors(cudaMemcpy(nodes,KERNEL_GPU_nodes,sizeof(int),cudaMemcpyDeviceToHost));
CudaErrors(cudaMemcpy(pruned_nodes,KERNEL_GPU_pruned_nodes,sizeof(int),cudaMemcpyDeviceToHost));

CudaErrors(cudaFree(GPUitem));
CudaErrors(cudaFree(KERNEL_GPU_nodes));
CudaErrors(cudaFree(KERNEL_GPU_pruned_nodes));

}

void GPU_pruning(struct BTree *item,struct nnb_info *GPUnnb,int npat,int rsize,int norm,int lpat,int *gpair1,double *gmin_dist,double *kernel_timer)
{

struct BTree *GPUitem;

struct nnb_info *KERNEL_GPUnnb;

int *block_index_minimum;
int *GPU_counter;
int *lpair1;
int *indices;

int counter=0;
double *block_distance_minimum;
double *lmin_dist;

CudaErrors(cudaMalloc((void**)&lpair1,npat*sizeof(int)));
CudaErrors(cudaMalloc((void**)&indices,npat*sizeof(int)));
CudaErrors(cudaMalloc((void**)&lmin_dist,npat*sizeof(double)));

cudaStream_t stream1,stream2;

cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

CudaErrors(cudaMalloc((void**)&GPUitem,npat*sizeof(BTree)));
CudaErrors(cudaMemcpy(GPUitem,item,npat*sizeof(BTree),cudaMemcpyHostToDevice));  

DATATYPE **rep,*rep1;
for(unsigned int i=0;i<npat;i++)
{

unsigned int maxrep=(item[i].size<=rsize)?item[i].size:rsize;

CudaErrors(cudaMalloc((void**)&rep,maxrep*sizeof(DATATYPE*)));

for(unsigned int j=0;j<maxrep;j++)
{

CudaErrors(cudaMalloc((void**)&rep1,lpat*sizeof(DATATYPE)));

CudaErrors(cudaMemcpy(rep1,item[i].rep[j],lpat*sizeof(DATATYPE),cudaMemcpyHostToDevice)); 
CudaErrors(cudaMemcpy(&rep[j],&rep1,sizeof(DATATYPE*),cudaMemcpyHostToDevice)); 
}

CudaErrors(cudaMemcpy(&(GPUitem[i].rep),&rep,sizeof(DATATYPE**),cudaMemcpyHostToDevice));

} 

CudaErrors(cudaMalloc((void**)&KERNEL_GPUnnb,npat*sizeof(nnb_info)));
CudaErrors(cudaMemcpy(KERNEL_GPUnnb,GPUnnb,npat*sizeof(nnb_info),cudaMemcpyHostToDevice));

CudaErrors(cudaMalloc((void**)&GPU_counter,sizeof(int)));
CudaErrors(cudaMemcpyAsync(GPU_counter,&counter,sizeof(int),cudaMemcpyHostToDevice,stream2));

gridsize=(npat+blocksize-1)/blocksize;

CudaErrors(cudaMalloc((void**)&block_index_minimum,gridsize*sizeof(int)));
CudaErrors(cudaMalloc((void**)&block_distance_minimum,gridsize*sizeof(double)));

CudaTimer timer;

CreateTimer(&timer);
StartTimer(&timer,stream1);        

GPU_update_nnb_kernel<<<gridsize,blocksize,0,stream1>>>(GPUitem,KERNEL_GPUnnb,npat,lpair1,lmin_dist);
CudaErrors(cudaStreamSynchronize(stream1));

StopTimer(&timer,stream1);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CreateTimer(&timer);
StartTimer(&timer,stream2);        

GPU_pruning_kernel1<<<gridsize,blocksize,0,stream2>>>(GPUitem,KERNEL_GPUnnb,npat,GPU_counter,indices);
CudaErrors(cudaStreamSynchronize(stream2));

StopTimer(&timer,stream2);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaMemcpyAsync(&counter,GPU_counter,sizeof(int),cudaMemcpyDeviceToHost,stream2));

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);

transfer_data_to_constant_memory(lpat,rsize,norm);

unsigned int gridsizes1=(counter+blocksize-1)/blocksize;

CreateTimer(&timer);
StartTimer(&timer,0); 

GPU_pruning_kernel2<<<gridsizes1,blocksize>>>(GPUitem,KERNEL_GPUnnb,indices,counter,lpair1,lmin_dist);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

counter=0;
CudaErrors(cudaMemcpy(GPU_counter,&counter,sizeof(int),cudaMemcpyHostToDevice));

CreateTimer(&timer);
StartTimer(&timer,0);  

GPU_update_kernel3<<<gridsize,blocksize>>>(lpair1,lmin_dist,block_index_minimum,block_distance_minimum,npat,GPU_counter);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaMemcpy(&counter,GPU_counter,sizeof(int),cudaMemcpyDeviceToHost));

unsigned int gridsizes5=(gridsize+blocksize-1)/blocksize+1;

CreateTimer(&timer);
StartTimer(&timer,0);
GPU_update_kernel4<<<gridsizes5,blocksize>>>(block_index_minimum,block_distance_minimum,gridsize,counter);
CudaErrors(cudaDeviceSynchronize());
StopTimer(&timer,0);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaFree(lpair1));
CudaErrors(cudaFree(lmin_dist));

CudaErrors(cudaFree(GPU_counter));

CudaErrors(cudaMemcpy(GPUnnb,KERNEL_GPUnnb,npat*sizeof(nnb_info),cudaMemcpyDeviceToHost));

if(gridsize>1)
{
CudaErrors(cudaFree(rep1));
CudaErrors(cudaFree(rep));
CudaErrors(cudaFree(GPUitem));

CudaErrors(cudaFree(KERNEL_GPUnnb));

if(counter>0) 
{  
if(gridsize>counter)
{
gridsize=counter;
}

while(gridsize>1)
{
new_gridsize=(gridsize+blocksize-1)/blocksize;

CreateTimer(&timer);
StartTimer(&timer,0);  


GPU_find_minimum_distance_pair<<<new_gridsize,blocksize>>>(block_index_minimum,block_distance_minimum,gridsize);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
*kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

gridsize=new_gridsize;
}

}

}else
{
CudaErrors(cudaFree(rep1));
CudaErrors(cudaFree(rep));
CudaErrors(cudaFree(GPUitem));

CudaErrors(cudaFree(KERNEL_GPUnnb));
}  

CudaErrors(cudaMemcpy(gpair1,block_index_minimum,sizeof(int),cudaMemcpyHostToHost));
CudaErrors(cudaMemcpy(gmin_dist,block_distance_minimum,sizeof(double),cudaMemcpyHostToHost));

CudaErrors(cudaFree(block_index_minimum));
CudaErrors(cudaFree(block_distance_minimum));

}

void Gpu_Cuda_Cure_Part_Results(int npat,int lpat,int csize,double init_timer,double find_mdp_timer,double clustering_timer,double merge_timer,double update_timer,double pruning_timer,const char *type)
{

struct Timers *BT=(Timers*)malloc(sizeof(Timers));
const char *path;
   
BT->init_timer=Timer_Array(1);
BT->find_mdp_timer=Timer_Array(1);
BT->clustering_timer=Timer_Array(1);
BT->merge_timer=Timer_Array(1);
BT->update_timer=Timer_Array(1);
BT->pruning_timer=Timer_Array(1);

BT->init_timer[0]=init_timer;
BT->find_mdp_timer[0]=find_mdp_timer;
BT->clustering_timer[0]=clustering_timer;
BT->merge_timer[0]=merge_timer;
BT->update_timer[0]=update_timer;
BT->pruning_timer[0]=pruning_timer;

char *previous_path=current_path();

if(strcmp(type,"SerialCure")==0)
{
path="/SerialCure";

}else if(strcmp(type,"CudaCurePart")==0)
{
path="/CudaCurePart";
}

path=ConcenateString(previous_path,path);  
change_directory(path);

show_time_results_for_each_case(type,BT,npat,lpat,csize);

change_directory(previous_path);

}

void Gpu_Cuda_Cure_Print_Results(struct BTree *item,int npat,int rsize,int lpat,int clusters,const char *type)
{

const char *path;
char *previous_path=current_path();

if(strcmp(type,"SerialCure")==0)
{
path="/SerialCure";

}else if(strcmp(type,"CudaCurePart")==0)
{
path="/CudaCurePart";
}

path=ConcenateString(previous_path,path);

change_directory(path);

print_results(type,item,npat,clusters,rsize,lpat);

change_directory(previous_path);

}

}