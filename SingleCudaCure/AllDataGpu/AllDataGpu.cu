#include "../../CudaCheckError/cudacheckerror.cuh"
#include "../../CudaTimer/cudatimer.cuh"
#include "../../CudaKernelsHostDeviceFunctions/KernelsHostDeviceFunctions.cuh"

extern "C"{

#include "../../CudaCure.h"

}
extern "C"
{

void GPU_Clustering(struct BTree *item,int npat,int rsize,int norm,int lpat,int csize)
{

printf("--------------------------------------------------------------ALL DATA GPU-------------------------------------------------------------------------\n\n");

CudaTimer timer,cluster_timer;

struct Timers *BT1=(Timers*)malloc(sizeof(Timers));

BT1->init_timer=Timer_Array(1);
BT1->find_mdp_timer=Timer_Array(1);
BT1->clustering_timer=Timer_Array(1);
BT1->merge_timer=Timer_Array(1);
BT1->update_timer=Timer_Array(1);
BT1->pruning_timer=Timer_Array(1);

cudaStream_t streams[4];

for(unsigned int i=0; i<4; i++)
{
CudaErrors(cudaStreamCreate(&streams[i]));
}

int nodes=npat;
int pruned_nodes=0;
int counter,counter1,pair1,pair2,node; 

struct BTree *GPUitem;   
struct BTree *GPU_merge_tmp_item;

struct nnb_info *KERNEL_GPUnnb; 

int *gpair1; 
int *temp_indices;  
int *GPU_counter; 
int *indices; 
int *lpair1; 
int *KERNEL_GPU_nodes;
int *KERNEL_GPU_pruned_nodes; 

double *gmin_dist;
double *lmin_dist;
double *min_dist; 
double *temp_distances;

gridsize=(npat+blocksize-1)/blocksize; // number of blocks

CudaErrors(cudaMalloc((void**)&GPU_counter,sizeof(int)));

CudaErrors(cudaMalloc((void**)&KERNEL_GPU_nodes,sizeof(int)));
CudaErrors(cudaMalloc((void**)&KERNEL_GPU_pruned_nodes,sizeof(int)));

CudaErrors(cudaMalloc((void**)&temp_indices,gridsize*sizeof(int))); 
CudaErrors(cudaMalloc((void**)&temp_distances,gridsize*sizeof(double)));

CudaErrors(cudaMalloc((void**)&gpair1,gridsize*sizeof(int)));
CudaErrors(cudaMalloc((void**)&gmin_dist,gridsize*sizeof(double)));

CudaErrors(cudaMalloc((void**)&indices,npat*sizeof(int)));
CudaErrors(cudaMalloc((void**)&lpair1,npat*sizeof(int)));
CudaErrors(cudaMalloc((void**)&lmin_dist,npat*sizeof(double)));

CudaErrors(cudaMalloc((void**)&min_dist,sizeof(double)));

CudaErrors(cudaMalloc((void**)&GPUitem,npat*sizeof(BTree)));
CudaErrors(cudaMemcpy(GPUitem,item,npat*sizeof(BTree),cudaMemcpyHostToDevice));

int **pats;
int *pats1;

DATATYPE **rep;
DATATYPE *rep1;
DATATYPE *mean;

for(unsigned int i=0;i<npat;i++)
{
CudaErrors(cudaMalloc((void**)&rep,rsize*sizeof(DATATYPE*)));

register int maxrep=(item[i].size<=rsize)?item[i].size:rsize;

for(register int j=0;j<maxrep;j++)
{
CudaErrors(cudaMalloc((void**)&rep1,lpat*sizeof(DATATYPE)));   
CudaErrors(cudaMemcpy(rep1,item[i].rep[j],lpat*sizeof(DATATYPE),cudaMemcpyHostToDevice)); 
CudaErrors(cudaMemcpy(&rep[j],&rep1,sizeof(DATATYPE*),cudaMemcpyHostToDevice)); 
}

if(maxrep<rsize)
{
for(register int j=maxrep;j<rsize;j++)
{  
CudaErrors(cudaMalloc((void**)&rep1,lpat*sizeof(DATATYPE)));
CudaErrors(cudaMemcpy(&rep[j],&rep1,sizeof(DATATYPE*),cudaMemcpyHostToDevice)); 
}
}

CudaErrors(cudaMemcpy(&(GPUitem[i].rep),&rep,sizeof(DATATYPE**),cudaMemcpyHostToDevice));

CudaErrors(cudaMalloc((void**)&mean,lpat*sizeof(DATATYPE)));
CudaErrors(cudaMemcpy(mean,item[i].mean,lpat*sizeof(DATATYPE),cudaMemcpyHostToDevice));
CudaErrors(cudaMemcpy(&(GPUitem[i].mean),&mean,sizeof(DATATYPE*),cudaMemcpyHostToDevice));
}

CudaErrors(cudaMalloc((void**)&pats,npat*sizeof(int*)));
for(unsigned int i=0;i<npat;i++)
{
CudaErrors(cudaMalloc((void**)&pats1,nnpc*sizeof(int)));   
CudaErrors(cudaMemcpy(pats1,item[i].pats,nnpc*sizeof(int),cudaMemcpyHostToDevice));
CudaErrors(cudaMemcpy(&(pats[i]),&pats1,sizeof(int*),cudaMemcpyHostToDevice));
}


CudaErrors(cudaMalloc((void**)&GPU_merge_tmp_item,sizeof(BTree)));

DATATYPE **rep2;
DATATYPE *rep3;

CudaErrors(cudaMalloc((void**)&rep2,(2*rsize)*sizeof(DATATYPE*)));

for(unsigned int i =0; i<(2*rsize); i++)
{
CudaErrors(cudaMalloc((void**)&rep3,lpat*sizeof(DATATYPE)));  
CudaErrors(cudaMemcpy(&rep2[i],&rep3,sizeof(DATATYPE*),cudaMemcpyHostToDevice)); 
}

CudaErrors(cudaMemcpy(&(GPU_merge_tmp_item[0].rep),&rep2,sizeof(DATATYPE*),cudaMemcpyHostToDevice));

CudaErrors(cudaMalloc((void**)&KERNEL_GPUnnb,npat*sizeof(nnb_info)));

one_minus_alpha_rev=one_minus_alpha_rev_value();

//-------------------------------------------------------- transfer data to constant memory------------------------------------------------------------------

transfer_data_to_constant_memory(alpha,one_minus_alpha_rev);
transfer_data_to_constant_memory(lpat,rsize,norm); 

//--------------------------------------------------------------------------------------------------------------------------------------------------------------

//-----------------------------------------------------------------initialization phase-------------------------------------------------------------------------

CreateTimer(&timer);
StartTimer(&timer,0);

GPU_initialization_phase1<<<gridsize,blocksize>>>(GPUitem,KERNEL_GPUnnb,npat);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
BT1->init_timer[0]=GetElapsedTime(&timer);
DestroyTimer(&timer);

//----------------------------------------------------------------------------------------------------------------------------------------------------------------- 

//------------------------------------------------------------------find nearest pair------------------------------------------------------------------------------

CreateTimer(&timer);
StartTimer(&timer,0);

NO_NONE_MDP<<<gridsize,blocksize>>>(GPUitem,KERNEL_GPUnnb,lpair1,lmin_dist,GPU_counter,npat);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
BT1->find_mdp_timer[0]=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaMemcpy(&counter,GPU_counter,sizeof(int),cudaMemcpyDeviceToHost));

gridsize=(counter+blocksize-1)/blocksize;

CreateTimer(&timer);
StartTimer(&timer,0);

GPU_find_minimum_distance_pair_per_block<<<gridsize,blocksize>>>(lpair1,lmin_dist,gpair1,gmin_dist,counter);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
BT1->find_mdp_timer[0]+GetElapsedTime(&timer);
DestroyTimer(&timer);


if(gridsize>1)
{
while(gridsize>1)
{
new_gridsize=(gridsize+blocksize-1)/blocksize;

CreateTimer(&timer);
StartTimer(&timer,0);

GPU_find_minimum_distance_pair<<<new_gridsize,blocksize>>>(gpair1,gmin_dist,gridsize);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
BT1->find_mdp_timer[0]+=GetElapsedTime(&timer);
DestroyTimer(&timer);

gridsize=new_gridsize;

}
}

//------------------------------------------------------------------------------------------------------------------------------------------------------------------

printf("ENTERING CLUSTERING PHASE (nodes=%d, clusters=%d)\n", nodes, csize); fflush(0);

CreateTimer(&cluster_timer);
StartTimer(&cluster_timer,0);

while(nodes>csize)
{

gridsize=(npat+blocksize-1)/blocksize;

CudaErrors(cudaMemcpy(&pair1,gpair1,sizeof(int),cudaMemcpyDeviceToHost));

if(pair1==NONE)break;

CudaErrors(cudaMemcpy(&pair2,&KERNEL_GPUnnb[pair1].index,sizeof(int),cudaMemcpyDeviceToHost));

GPUroot<<<1,1>>>(min_dist,gmin_dist);
CudaErrors(cudaDeviceSynchronize());

//-------------------------------------------------------------------------------merge----------------------------------------------------------------------------- 
CreateTimer(&timer);
StartTimer(&timer,0);
GPUmerge<<<1,1>>>(GPUitem,GPU_merge_tmp_item,pats,pair1,pair2,min_dist,size_limit);
CudaErrors(cudaDeviceSynchronize());
StopTimer(&timer,0);
BT1->merge_timer[0]+=GetElapsedTime(&timer);
DestroyTimer(&timer);

/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  

/*--------------------------------------------------------------------------------update----------------------------------------------------------------------------------------------*/

counter=0;

CreateTimer(&timer);
StartTimer(&timer,streams[0]);
GPU_update_nnb_kernel<<<(pair2+blocksize-1)/blocksize,blocksize,0,streams[0]>>>(GPUitem,KERNEL_GPUnnb,pair2,lpair1,lmin_dist);
CudaErrors(cudaStreamSynchronize(streams[0]));
StopTimer(&timer,streams[0]);
BT1->update_timer[0]+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CreateTimer(&timer);
StartTimer(&timer,streams[1]);
GPU_update_nnb_kernel1<<<((pair1-pair2)+blocksize-1)/blocksize,blocksize,0,streams[1]>>>(GPUitem,KERNEL_GPUnnb,pair1,pair1,pair2,lpair1,lmin_dist,pair2);
CudaErrors(cudaStreamSynchronize(streams[1]));
StopTimer(&timer,streams[1]);
BT1->update_timer[0]+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CreateTimer(&timer);
StartTimer(&timer,streams[2]);
GPU_update_nnb_kernel2<<<((npat-pair1)+blocksize-1)/blocksize,blocksize,0,streams[2]>>>(GPUitem,KERNEL_GPUnnb,npat,pair1,pair2,lpair1,lmin_dist,pair1);
CudaErrors(cudaStreamSynchronize(streams[2]));
StopTimer(&timer,streams[2]);
BT1->update_timer[0]+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaMemcpy(GPU_counter,&counter,sizeof(int),cudaMemcpyHostToDevice));

CreateTimer(&timer);
StartTimer(&timer,streams[3]);
GPU_update_nnb_kernel3<<<((npat-pair2)+blocksize-1)/blocksize,blocksize,0,streams[3]>>>(GPUitem,KERNEL_GPUnnb,npat,pair1,pair2,pair2,GPU_counter,indices);
CudaErrors(cudaStreamSynchronize(streams[3]));
StopTimer(&timer,streams[3]);
BT1->update_timer[0]+=GetElapsedTime(&timer);
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
BT1->update_timer[0]+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaMemcpy(&counter1,GPU_counter,sizeof(int),cudaMemcpyDeviceToHost));

CreateTimer(&timer);
StartTimer(&timer,0);
GPU_update_kernel4<<<((((node+blocksize-1)/blocksize)+1)+blocksize-1)/blocksize+1,blocksize>>>(temp_indices,temp_distances,(node+blocksize-1)/blocksize+1,counter1);
StopTimer(&timer,0);
BT1->update_timer[0]+=GetElapsedTime(&timer);
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
BT1->update_timer[0]+=GetElapsedTime(&timer);
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
BT1->update_timer[0]+=GetElapsedTime(&timer);
DestroyTimer(&timer);

}

counter=0;
CudaErrors(cudaMemcpy(GPU_counter,&counter,sizeof(int),cudaMemcpyHostToDevice));

CreateTimer(&timer);
StartTimer(&timer,0);

GPU_update_kernel3<<<gridsize,blocksize>>>(lpair1,lmin_dist,gpair1,gmin_dist,npat,GPU_counter);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
BT1->update_timer[0]+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaMemcpy(&counter,GPU_counter,sizeof(int),cudaMemcpyDeviceToHost));

CreateTimer(&timer);
StartTimer(&timer,0);
GPU_update_kernel4<<<(gridsize+blocksize-1)/blocksize+1,blocksize>>>(gpair1,gmin_dist,gridsize,counter);
StopTimer(&timer,0);
BT1->update_timer[0]+=GetElapsedTime(&timer);
DestroyTimer(&timer);


if((npat+blocksize-1)/blocksize>1)
{

if(counter>0)
{
unsigned int gridsize = (npat+blocksize-1)/blocksize;
if(gridsize>counter)
{
gridsize=counter;
}

while(gridsize>1)
{
new_gridsize=(gridsize+blocksize-1)/blocksize;

CreateTimer(&timer);
StartTimer(&timer,0);

GPU_find_minimum_distance_pair<<<new_gridsize,blocksize>>>(gpair1,gmin_dist,gridsize);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
BT1->update_timer[0]+=GetElapsedTime(&timer);
DestroyTimer(&timer);      

gridsize=new_gridsize;
}

} 

}

//----------------------------------------------------------------------------------------------------------------------------------------------------------------

--nodes;

/*---------------------------------------------------------------- prune clusters--------------------------------------------------------------------------------*/   
if(prune_clusters)
{

if (nodes == (int)(npat * FirstPruneRatio)) 
{

CudaErrors(cudaMemcpy(KERNEL_GPU_nodes,&nodes,sizeof(int),cudaMemcpyHostToDevice));
CudaErrors(cudaMemcpy(KERNEL_GPU_pruned_nodes,&pruned_nodes,sizeof(int),cudaMemcpyHostToDevice));

double kernel_timer=0;

gridsize=(npat+blocksize-1)/blocksize;

printf("==== First phase of pruning at %d nodes remaining ====\n", nodes);

CreateTimer(&timer);
StartTimer(&timer,0);

GPU_first_pruning_kernel<<<gridsize,blocksize>>>(GPUitem,npat,KERNEL_GPU_nodes,KERNEL_GPU_pruned_nodes);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
kernel_timer=GetElapsedTime(&timer);
BT1->pruning_timer[0]+=kernel_timer;
DestroyTimer(&timer);

CudaErrors(cudaMemcpy(&nodes,KERNEL_GPU_nodes,sizeof(int),cudaMemcpyDeviceToHost));
CudaErrors(cudaMemcpy(&pruned_nodes,KERNEL_GPU_pruned_nodes,sizeof(int),cudaMemcpyDeviceToHost));

printf("GPU first pruning - (1): %lf seconds (pruned nodes = %d)\n",kernel_timer, pruned_nodes);

kernel_timer=0;

counter=0;
CudaErrors(cudaMemcpy(GPU_counter,&counter,sizeof(int),cudaMemcpyHostToDevice));

CreateTimer(&timer);
StartTimer(&timer,streams[0]);

GPU_update_nnb_kernel<<<gridsize,blocksize,0,streams[0]>>>(GPUitem,KERNEL_GPUnnb,npat,lpair1,lmin_dist);
CudaErrors(cudaStreamSynchronize(streams[0]));

StopTimer(&timer,streams[0]);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CreateTimer(&timer);
StartTimer(&timer,streams[1]);

GPU_pruning_kernel1<<<gridsize,blocksize,0,streams[1]>>>(GPUitem,KERNEL_GPUnnb,npat,GPU_counter,indices);
CudaErrors(cudaStreamSynchronize(streams[1]));

StopTimer(&timer,streams[1]);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaMemcpy(&counter,GPU_counter,sizeof(int),cudaMemcpyDeviceToHost));

CreateTimer(&timer);
StartTimer(&timer,0);  

GPU_pruning_kernel2<<<(counter+blocksize-1)/blocksize,blocksize>>>(GPUitem,KERNEL_GPUnnb,indices,counter,lpair1,lmin_dist);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer); 

counter=0;
CudaErrors(cudaMemcpy(GPU_counter,&counter,sizeof(int),cudaMemcpyHostToDevice));

CreateTimer(&timer);
StartTimer(&timer,0);   

GPU_update_kernel3<<<gridsize,blocksize>>>(lpair1,lmin_dist,gpair1,gmin_dist,npat,GPU_counter);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);   

CudaErrors(cudaMemcpy(&counter,GPU_counter,sizeof(int),cudaMemcpyDeviceToHost)); 

unsigned int gridsizes5=(gridsize+blocksize-1)/blocksize+1;

CreateTimer(&timer);
StartTimer(&timer,0);
GPU_update_kernel4<<<gridsizes5,blocksize>>>(gpair1,gmin_dist,gridsize,counter);
StopTimer(&timer,0);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer); 

if(gridsize>1)
{  
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

GPU_find_minimum_distance_pair<<<new_gridsize,blocksize>>>(gpair1,gmin_dist,gridsize);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

gridsize=new_gridsize;
}
}
}

printf("GPU first pruning - (2): %lf seconds\n",kernel_timer);
BT1->pruning_timer[0]+=kernel_timer;

}else if(nodes == csize * SecondPruneMulti)
{       
CudaErrors(cudaMemcpy(KERNEL_GPU_nodes,&nodes,sizeof(int),cudaMemcpyHostToDevice));
CudaErrors(cudaMemcpy(KERNEL_GPU_pruned_nodes,&pruned_nodes,sizeof(int),cudaMemcpyHostToDevice));

double kernel_timer=0;

gridsize=(npat+blocksize-1)/blocksize;

printf("==== Second phase of pruning at %d nodes remaining ====\n", nodes);

CreateTimer(&timer);
StartTimer(&timer,0);

GPU_second_pruning_kernel<<<gridsize,blocksize>>>(GPUitem,npat,KERNEL_GPU_nodes,KERNEL_GPU_pruned_nodes);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
kernel_timer=GetElapsedTime(&timer);
BT1->pruning_timer[0]+=kernel_timer;
DestroyTimer(&timer);

CudaErrors(cudaMemcpy(&nodes,KERNEL_GPU_nodes,sizeof(int),cudaMemcpyDeviceToHost));
CudaErrors(cudaMemcpy(&pruned_nodes,KERNEL_GPU_pruned_nodes,sizeof(int),cudaMemcpyDeviceToHost));

printf("GPU second pruning - (1): %lf seconds - pruned nodes = %d\n",kernel_timer, pruned_nodes);

kernel_timer=0;

counter=0;
CudaErrors(cudaMemcpy(GPU_counter,&counter,sizeof(int),cudaMemcpyHostToDevice));

CreateTimer(&timer);
StartTimer(&timer,streams[0]);

GPU_update_nnb_kernel<<<gridsize,blocksize,0,streams[0]>>>(GPUitem,KERNEL_GPUnnb,npat,lpair1,lmin_dist);
CudaErrors(cudaStreamSynchronize(streams[0]));

StopTimer(&timer,streams[0]);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CreateTimer(&timer);
StartTimer(&timer,streams[1]);

GPU_pruning_kernel1<<<gridsize,blocksize,0,streams[1]>>>(GPUitem,KERNEL_GPUnnb,npat,GPU_counter,indices);
CudaErrors(cudaStreamSynchronize(streams[1]));

StopTimer(&timer,streams[1]);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CudaErrors(cudaMemcpy(&counter,GPU_counter,sizeof(int),cudaMemcpyDeviceToHost));

CreateTimer(&timer);
StartTimer(&timer,0);  

GPU_pruning_kernel2<<<(counter+blocksize-1)/blocksize,blocksize>>>(GPUitem,KERNEL_GPUnnb,indices,counter,lpair1,lmin_dist);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer); 

counter=0;
CudaErrors(cudaMemcpy(GPU_counter,&counter,sizeof(int),cudaMemcpyHostToDevice));

CreateTimer(&timer);
StartTimer(&timer,0);   

GPU_update_kernel3<<<gridsize,blocksize>>>(lpair1,lmin_dist,gpair1,gmin_dist,npat,GPU_counter);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);   

CudaErrors(cudaMemcpy(&counter,GPU_counter,sizeof(int),cudaMemcpyDeviceToHost)); 

unsigned int gridsizes5=(gridsize+blocksize-1)/blocksize+1;

CreateTimer(&timer);
StartTimer(&timer,0);

GPU_update_kernel4<<<gridsizes5,blocksize>>>(gpair1,gmin_dist,gridsize,counter);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer); 

if(gridsize>1)
{
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

GPU_find_minimum_distance_pair<<<new_gridsize,blocksize>>>(gpair1,gmin_dist,gridsize);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

gridsize=new_gridsize;
}


}

}

printf("GPU second pruning - (2): %lf seconds\n",kernel_timer);

BT1->pruning_timer[0]+=kernel_timer;

}     
}   
} 

StopTimer(&cluster_timer,0);
BT1->clustering_timer[0]=GetElapsedTime(&cluster_timer);
DestroyTimer(&cluster_timer);

char *previous_path=current_path();
const char *path="/SingleCudaCure/AllDataGpu";
path=ConcenateString(previous_path,path);

change_directory(path);
show_time_results_for_each_case("AllDataGpu",BT1,npat,lpat,csize);

struct BTree *item1=alloc_item(item,nodes,rsize,lpat);
if(item1==NULL)
{
exit(1);
}

int root;
int pruned;

unsigned int k=0;

for(unsigned int i=0;i<npat;i++)
{
CudaErrors(cudaMemcpy(&root,&GPUitem[i].root,sizeof(int),cudaMemcpyDeviceToHost));
CudaErrors(cudaMemcpy(&pruned,&GPUitem[i].pruned,sizeof(int),cudaMemcpyDeviceToHost));

if(root==TRUE && pruned== FALSE)
{
CudaErrors(cudaMemcpy(&item1[k].root,&GPUitem[i].root,sizeof(int),cudaMemcpyDeviceToHost));
CudaErrors(cudaMemcpy(&item1[k].pruned,&GPUitem[i].pruned,sizeof(int),cudaMemcpyDeviceToHost)); 
CudaErrors(cudaMemcpy(&item1[k].full,&GPUitem[i].full,sizeof(int),cudaMemcpyDeviceToHost));
CudaErrors(cudaMemcpy(&item1[k].size,&GPUitem[i].size,sizeof(int),cudaMemcpyDeviceToHost));

CudaErrors(cudaMemcpy(&item1[k].distance,&GPUitem[i].distance,sizeof(double),cudaMemcpyDeviceToHost));

CudaErrors(cudaMalloc((void**)&rep1,(rsize*lpat)*sizeof(DATATYPE)));
CudaErrors(cudaMalloc((void**)&mean,lpat*sizeof(DATATYPE)));

dim3 blockDim(16,16);
dim3 gridDim((rsize+blockDim.x-1)/blockDim.x,(lpat+blockDim.y-1)/blockDim.y);

BTree_transfer<<<gridDim,blockDim>>>(&GPUitem[i],rep1,mean,rsize,lpat);
cudaDeviceSynchronize();

DATATYPE *rep2=(DATATYPE*)malloc((rsize*lpat)*sizeof(DATATYPE));
if(rep2==NULL)
{
exit(1);
}

CudaErrors(cudaMemcpy(rep2,rep1,(rsize*lpat)*sizeof(DATATYPE),cudaMemcpyDeviceToHost));

register int maxrep=(item1[k].size<=rsize)?item1[k].size:rsize;

for(register int j=0;j<maxrep;j++)
{
memcpy(item1[k].rep[j],&rep2[j*lpat],lpat*sizeof(DATATYPE));    
}

CudaErrors(cudaMemcpy(item1[k].mean,mean,lpat*sizeof(DATATYPE),cudaMemcpyDeviceToHost));

CudaErrors(cudaMalloc((void**)&pats1,nnpc*sizeof(int)));
pats_transfer<<<(nnpc+blocksize-1)/blocksize,blocksize>>>(&pats[i],pats1);
cudaDeviceSynchronize();
CudaErrors(cudaMemcpy(item1[k].pats,pats1,nnpc*sizeof(int),cudaMemcpyDeviceToHost)); 

k++;
}
}

if(print_clusters)
{
print_results("AllDataGpu",item1,npat,k,rsize,lpat);
}

change_directory(previous_path);

FreeItems(item1,nodes,rsize);

TimersFree(BT1); 

for(unsigned int i = 0; i<4; i++)
{
CudaErrors(cudaStreamDestroy(streams[i]));
}

CudaErrors(cudaFree(lpair1));
CudaErrors(cudaFree(lmin_dist));
CudaErrors(cudaFree(indices));
CudaErrors(cudaFree(KERNEL_GPU_nodes));
CudaErrors(cudaFree(KERNEL_GPU_pruned_nodes));
CudaErrors(cudaFree(gpair1));
CudaErrors(cudaFree(gmin_dist));
CudaErrors(cudaFree(temp_indices));
CudaErrors(cudaFree(temp_distances));  
CudaErrors(cudaFree(GPU_counter)); 
CudaErrors(cudaFree(KERNEL_GPUnnb));
CudaErrors(cudaFree(rep2));
CudaErrors(cudaFree(rep3));
CudaErrors(cudaFree(GPU_merge_tmp_item));
CudaErrors(cudaFree(pats1));
CudaErrors(cudaFree(pats));
CudaErrors(cudaFree(mean));
CudaErrors(cudaFree(rep1));
CudaErrors(cudaFree(rep));
CudaErrors(cudaFree(GPUitem));

}

}
