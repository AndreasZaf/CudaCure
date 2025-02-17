#include "../../CudaCheckError/cudacheckerror.cuh"
#include "../../CudaTimer/cudatimer.cuh"
#include "../../CudaKernelsHostDeviceFunctions/KernelsHostDeviceFunctions.cuh"

extern "C"
{
#include "../../CudaCure.h"
}


extern "C"
{

void GPU_Clustering_Pinned_Memory_CMH(struct BTree *item,int npat,int rsize,int norm,int lpat,int csize)
{

printf("--------------------------------------------------------------PINNED MEMORY CMH-------------------------------------------------------------------------\n\n");

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

struct BTree *GPUitem;
struct BTree *GPU_merge_tmp_item;

struct nnb_info *KERNEL_GPUnnb;

int pair1,pair2;

int *gpair1;
int *temp_indices;
int *nodes;
int *pruned_nodes;
int *counter;
int *counter1;
int *indices;
int *lpair1;

double *gmin_dist;
double *lmin_dist;
double *min_dist;
double *temp_distances;

gridsize=(npat+blocksize-1)/blocksize;

CudaErrors(cudaMallocHost((void**)&gpair1,gridsize*sizeof(int)));
CudaErrors(cudaMallocHost((void**)&gmin_dist,gridsize*sizeof(double)));

CudaErrors(cudaMallocHost((void**)&temp_indices,gridsize*sizeof(int)));
CudaErrors(cudaMallocHost((void**)&temp_distances,gridsize*sizeof(double)));

CudaErrors(cudaMallocHost((void**)&nodes,sizeof(int)));
CudaErrors(cudaMallocHost((void**)&pruned_nodes,sizeof(int)));
*pruned_nodes=0;
*nodes=npat;

CudaErrors(cudaMallocHost((void**)&counter,sizeof(int)));
CudaErrors(cudaMallocHost((void**)&counter1,sizeof(int)));

CudaErrors(cudaMallocHost((void**)&indices,npat*sizeof(int)));
CudaErrors(cudaMallocHost((void**)&lpair1,npat*sizeof(int)));
CudaErrors(cudaMallocHost((void**)&lmin_dist,npat*sizeof(double)));

CudaErrors(cudaMallocHost((void**)&min_dist,sizeof(double)));

/*-----------------------------------------------------------------------------GPUitem---------------------------------------------------------------------------------*/

CudaErrors(cudaMallocHost((void**)&GPUitem,npat*sizeof(BTree)));
CudaErrors(cudaMemcpy(GPUitem,item,npat*sizeof(BTree),cudaMemcpyHostToDevice));


int **pats,*pats1;
DATATYPE **rep,*rep1,*mean;

for(unsigned int i=0;i<npat; i++)
{


CudaErrors(cudaMallocHost((void**)&rep,rsize*sizeof(DATATYPE*)));

register int maxrep=(item[i].size<=rsize)?item[i].size:rsize;

for(register int j=0;j<maxrep;j++)
{

CudaErrors(cudaMallocHost((void**)&rep1,lpat*sizeof(DATATYPE)));

CudaErrors(cudaMemcpy(rep1,item[i].rep[j],lpat*sizeof(DATATYPE),cudaMemcpyHostToDevice)); 
CudaErrors(cudaMemcpy(&rep[j],&rep1,sizeof(DATATYPE*),cudaMemcpyHostToDevice)); 
}

if(maxrep<rsize)
{

for(register int j=maxrep;j<rsize;j++)
{

CudaErrors(cudaMallocHost((void**)&rep1,lpat*sizeof(DATATYPE)));
CudaErrors(cudaMemcpy(&rep[j],&rep1,sizeof(DATATYPE*),cudaMemcpyHostToDevice)); 
}
}  
CudaErrors(cudaMemcpy(&(GPUitem[i].rep),&rep,sizeof(DATATYPE**),cudaMemcpyHostToDevice));

CudaErrors(cudaMallocHost((void**)&mean,lpat*sizeof(DATATYPE)));
CudaErrors(cudaMemcpy(mean,item[i].mean,lpat*sizeof(DATATYPE),cudaMemcpyHostToDevice));
CudaErrors(cudaMemcpy(&(GPUitem[i].mean),&mean,sizeof(DATATYPE*),cudaMemcpyHostToDevice));

}

CudaErrors(cudaMallocHost((void**)&pats,npat*sizeof(int*)));
for(unsigned int i=0;i<npat;i++)
{
CudaErrors(cudaMallocHost((void**)&pats1,nnpc*sizeof(int)));   
CudaErrors(cudaMemcpy(pats1,item[i].pats,nnpc*sizeof(int),cudaMemcpyHostToDevice));
CudaErrors(cudaMemcpy(&(pats[i]),&pats1,sizeof(int*),cudaMemcpyHostToDevice));
}


/*-----------------------------------------------------------------------GPU_merge_tmp_item-----------------------------------------------------------------------*/

CudaErrors(cudaMallocHost((void**)&GPU_merge_tmp_item,sizeof(BTree)));

DATATYPE **rep2,*rep3;
CudaErrors(cudaMallocHost((void**)&rep2,(2*rsize)*sizeof(DATATYPE*)));

for(unsigned int i =0; i<(2*rsize); i++)
{
CudaErrors(cudaMallocHost((void**)&rep3,lpat*sizeof(DATATYPE)));  
CudaErrors(cudaMemcpy(&rep2[i],&rep3,sizeof(DATATYPE*),cudaMemcpyHostToDevice)); 
}

CudaErrors(cudaMemcpy(&(GPU_merge_tmp_item[0].rep),&rep2,sizeof(DATATYPE*),cudaMemcpyHostToDevice));


/*-----------------------------------------------------------------------KERNEL_GPUnnb-----------------------------------------------------------------------------------*/
CudaErrors(cudaMallocHost((void**)&KERNEL_GPUnnb,npat*sizeof(nnb_info)));

/*-------------------------------------------------------constant memory-----------------------------------------------------------------------------*/
one_minus_alpha_rev=one_minus_alpha_rev_value();
transfer_data_to_constant_memory(alpha,one_minus_alpha_rev);
transfer_data_to_constant_memory(lpat,rsize,norm);     

/*--------------------------------------------------------------------------------------------------------------------------------------------------------*/

/*------------------------------------------------initialization phase-------------------------------------------------------------------*/



CreateTimer(&timer);
StartTimer(&timer,0);

GPU_initialization_phase1<<<gridsize,blocksize>>>(GPUitem,KERNEL_GPUnnb,npat);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
BT1->init_timer[0]=GetElapsedTime(&timer);
DestroyTimer(&timer);



/*------------------------------------------------------------------------------------------------------------------------------------------*/

/*--------------------------------------------------minimum distance pair--------------------------------------------------------------------*/



CreateTimer(&timer);
StartTimer(&timer,0);

NO_NONE_MDP<<<gridsize,blocksize>>>(GPUitem,KERNEL_GPUnnb,lpair1,lmin_dist,counter,npat);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
BT1->find_mdp_timer[0]=GetElapsedTime(&timer);
DestroyTimer(&timer);

gridsize=(*counter+blocksize-1)/blocksize;

CreateTimer(&timer);
StartTimer(&timer,0);

GPU_find_minimum_distance_pair_per_block<<<gridsize,blocksize>>>(lpair1,lmin_dist,gpair1,gmin_dist,*counter);
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





/*--------------------------------------------------------------------------------------------------------------------------------------------*/

printf("ENTERING CLUSTERING PHASE (nodes=%d, clusters=%d)\n", *nodes, csize); fflush(0);

CreateTimer(&cluster_timer);
StartTimer(&cluster_timer,0);

while(*nodes>csize)
{

pair1=gpair1[0];

if(pair1==NONE) break;

gridsize=(npat+blocksize-1)/blocksize;

pair2=KERNEL_GPUnnb[pair1].index;

GPUroot<<<1,1>>>(min_dist,gmin_dist);
CudaErrors(cudaDeviceSynchronize());


/*--------------------------------------------------------merge-------------------------------------------------------------------------------------------*/




CreateTimer(&timer);
StartTimer(&timer,0);
GPUmerge<<<1,1>>>(GPUitem,GPU_merge_tmp_item,pats,pair1,pair2,min_dist,size_limit);
CudaErrors(cudaDeviceSynchronize());
StopTimer(&timer,0);
BT1->merge_timer[0]+=GetElapsedTime(&timer);
DestroyTimer(&timer);

/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  

/*-------------------------------------------------------update nnb----------------------------------------------------------------------------------------------*/

*counter=0;

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

CreateTimer(&timer);
StartTimer(&timer,streams[3]);
GPU_update_nnb_kernel3<<<((npat-pair2)+blocksize-1)/blocksize,blocksize,0,streams[3]>>>(GPUitem,KERNEL_GPUnnb,npat,pair1,pair2,pair2,counter,indices);
CudaErrors(cudaStreamSynchronize(streams[3]));
StopTimer(&timer,streams[3]);
BT1->update_timer[0]+=GetElapsedTime(&timer);
DestroyTimer(&timer);

for(unsigned int i=0;i<*counter;i++)
{

*counter1=0;

CreateTimer(&timer);
StartTimer(&timer,0);

GPU_update_kernel<<<(indices[i]+blocksize-1)/blocksize+1,blocksize>>>(GPUitem,indices[i],temp_indices,temp_distances,counter1);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
BT1->update_timer[0]+=GetElapsedTime(&timer);
DestroyTimer(&timer);


CreateTimer(&timer);
StartTimer(&timer,0);
GPU_update_kernel4<<<((((indices[i]+blocksize-1)/blocksize)+1)+blocksize-1)/blocksize+1,blocksize>>>(temp_indices,temp_distances,(indices[i]+blocksize-1)/blocksize+1,*counter1);
StopTimer(&timer,0);
BT1->update_timer[0]+=GetElapsedTime(&timer);
DestroyTimer(&timer);

if(((indices[i]+blocksize-1)/blocksize+1)>1)
{

if(*counter1>0)
{
unsigned int gridsizes5=(indices[i]+blocksize-1)/blocksize+1;

if(gridsizes5>*counter1)
{
gridsizes5=*counter1;
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

GPU_update_kernel2<<<1,1>>>(GPUitem,KERNEL_GPUnnb,indices[i],temp_indices,temp_distances,lpair1,lmin_dist);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
BT1->update_timer[0]+=GetElapsedTime(&timer);
DestroyTimer(&timer);

}

*counter=0;

CreateTimer(&timer);
StartTimer(&timer,0);

GPU_update_kernel3<<<gridsize,blocksize>>>(lpair1,lmin_dist,gpair1,gmin_dist,npat,counter);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
BT1->update_timer[0]+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CreateTimer(&timer);
StartTimer(&timer,0);
GPU_update_kernel4<<<(gridsize+blocksize-1)/blocksize+1,blocksize>>>(gpair1,gmin_dist,gridsize,*counter);
StopTimer(&timer,0);
BT1->update_timer[0]+=GetElapsedTime(&timer);
DestroyTimer(&timer);

if((npat+blocksize-1)/blocksize>1)
{

if(*counter>0)
{
unsigned int gridsize = (npat+blocksize-1)/blocksize;
if(gridsize>*counter)
{
gridsize=*counter;
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




/*--------------------------------------------------------------------------------------------------------------------------------------------------------------*/


--(*nodes);

/*---------------------------------------------------------------- prune clusters--------------------------------------------------------------------------------*/   
if(prune_clusters)
{

double kernel_timer;

if (*nodes == (int)(npat * FirstPruneRatio)) 
{

kernel_timer=0;

gridsize=(npat+blocksize-1)/blocksize;

printf("==== First phase of pruning at %d nodes remaining ====\n", *nodes);

CreateTimer(&timer);
StartTimer(&timer,0);

GPU_first_pruning_kernel<<<gridsize,blocksize>>>(GPUitem,npat,nodes,pruned_nodes);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
kernel_timer=GetElapsedTime(&timer);
BT1->pruning_timer[0]+=kernel_timer;
DestroyTimer(&timer);

printf("GPU first pruning - (1): %lf seconds (pruned nodes = %d)\n",kernel_timer, *pruned_nodes);

kernel_timer=0;

*counter=0;

CreateTimer(&timer);
StartTimer(&timer,streams[0]);

GPU_update_nnb_kernel<<<gridsize,blocksize,0,streams[0]>>>(GPUitem,KERNEL_GPUnnb,npat,lpair1,lmin_dist);
CudaErrors(cudaStreamSynchronize(streams[0]));

StopTimer(&timer,streams[0]);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CreateTimer(&timer);
StartTimer(&timer,streams[1]);

GPU_pruning_kernel1<<<gridsize,blocksize,0,streams[1]>>>(GPUitem,KERNEL_GPUnnb,npat,counter,indices);
CudaErrors(cudaStreamSynchronize(streams[1]));

StopTimer(&timer,streams[1]);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CreateTimer(&timer);
StartTimer(&timer,0);  

GPU_pruning_kernel2<<<(*counter+blocksize-1)/blocksize,blocksize>>>(GPUitem,KERNEL_GPUnnb,indices,*counter,lpair1,lmin_dist);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer); 

*counter=0;

CreateTimer(&timer);
StartTimer(&timer,0);   

GPU_update_kernel3<<<gridsize,blocksize>>>(lpair1,lmin_dist,gpair1,gmin_dist,npat,counter);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);   

unsigned int gridsizes5=(gridsize+blocksize-1)/blocksize+1;

CreateTimer(&timer);
StartTimer(&timer,0);
GPU_update_kernel4<<<gridsizes5,blocksize>>>(gpair1,gmin_dist,gridsize,*counter);
StopTimer(&timer,0);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer); 

if(gridsize>1)
{


if(*counter>0) 
{  
if(gridsize>*counter)
{
gridsize=*counter;
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





}else if(*nodes == csize * SecondPruneMulti)
{
kernel_timer=0;

gridsize=(npat+blocksize-1)/blocksize;

printf("==== Second phase of pruning at %d nodes remaining ====\n", *nodes);

CreateTimer(&timer);
StartTimer(&timer,0);
GPU_second_pruning_kernel<<<gridsize,blocksize>>>(GPUitem,npat,nodes,pruned_nodes);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
kernel_timer=GetElapsedTime(&timer);
BT1->pruning_timer[0]+=kernel_timer;
DestroyTimer(&timer);

printf("GPU second pruning - (1): %lf seconds - pruned nodes = %d\n",kernel_timer, *pruned_nodes);

kernel_timer=0;

*counter=0;

CreateTimer(&timer);
StartTimer(&timer,streams[0]);

GPU_update_nnb_kernel<<<gridsize,blocksize,0,streams[0]>>>(GPUitem,KERNEL_GPUnnb,npat,lpair1,lmin_dist);
CudaErrors(cudaStreamSynchronize(streams[0]));

StopTimer(&timer,streams[0]);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);

CreateTimer(&timer);
StartTimer(&timer,streams[1]);

GPU_pruning_kernel1<<<gridsize,blocksize,0,streams[1]>>>(GPUitem,KERNEL_GPUnnb,npat,counter,indices);
CudaErrors(cudaStreamSynchronize(streams[1]));

StopTimer(&timer,streams[1]);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);


CreateTimer(&timer);
StartTimer(&timer,0);  

GPU_pruning_kernel2<<<(*counter+blocksize-1)/blocksize,blocksize>>>(GPUitem,KERNEL_GPUnnb,indices,*counter,lpair1,lmin_dist);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer); 

*counter=0;

CreateTimer(&timer);
StartTimer(&timer,0);   

GPU_update_kernel3<<<gridsize,blocksize>>>(lpair1,lmin_dist,gpair1,gmin_dist,npat,counter);
CudaErrors(cudaDeviceSynchronize());

StopTimer(&timer,0);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);  

unsigned int gridsizes5=(gridsize+blocksize-1)/blocksize+1;

CreateTimer(&timer);
StartTimer(&timer,0);
GPU_update_kernel4<<<gridsizes5,blocksize>>>(gpair1,gmin_dist,gridsize,*counter);
StopTimer(&timer,0);
kernel_timer+=GetElapsedTime(&timer);
DestroyTimer(&timer);  

if(gridsize>1)
{


if(*counter>0) 
{  
if(gridsize>*counter)
{
gridsize=*counter;
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



/*------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/   

}

StopTimer(&cluster_timer,0);
BT1->clustering_timer[0]=GetElapsedTime(&cluster_timer);
DestroyTimer(&cluster_timer);

for(unsigned int i=0;i<npat;i++)
{

CudaErrors(cudaMemcpy(&(GPUitem[i].pats),&pats[i],sizeof(int*),cudaMemcpyHostToDevice)); 
}

char *previous_path=current_path();
const char *path="/SingleCudaCure/PinnedMemoryCMH";
path=ConcenateString(previous_path,path);

change_directory(path);
show_time_results_for_each_case("PinnedMemoryCMH",BT1,npat,lpat,csize);

if(print_clusters)
{
print_results("PinnedMemoryCMH",GPUitem,npat,*nodes,rsize,lpat);
}

change_directory(previous_path);

TimersFree(BT1);

CudaErrors(cudaFreeHost(indices));
CudaErrors(cudaFreeHost(pruned_nodes));
CudaErrors(cudaFreeHost(nodes));
CudaErrors(cudaFreeHost(lpair1));
CudaErrors(cudaFreeHost(lmin_dist));
CudaErrors(cudaFreeHost(counter));
CudaErrors(cudaFreeHost(counter1));
CudaErrors(cudaFreeHost(gpair1));
CudaErrors(cudaFreeHost(gmin_dist));
CudaErrors(cudaFreeHost(temp_indices));
CudaErrors(cudaFreeHost(temp_distances));

for(unsigned int i = 0; i<4; i++)
{
CudaErrors(cudaStreamDestroy(streams[i]));
}
CudaErrors(cudaFreeHost(KERNEL_GPUnnb));

CudaErrors(cudaFreeHost(rep2));
CudaErrors(cudaFreeHost(rep3));
CudaErrors(cudaFreeHost(GPU_merge_tmp_item));

CudaErrors(cudaFreeHost(pats1));
CudaErrors(cudaFreeHost(pats));
CudaErrors(cudaFreeHost(mean));

CudaErrors(cudaFreeHost(rep1));
CudaErrors(cudaFreeHost(rep));
CudaErrors(cudaFreeHost(GPUitem));


}


}