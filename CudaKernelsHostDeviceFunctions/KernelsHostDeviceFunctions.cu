#include "KernelsHostDeviceFunctions.cuh"

__constant__ int constant_lpat,constant_rsize,constant_norm;
__constant__ DATATYPE constant_alpha,constant_one_minus_alpha_rev;

__host__ DATATYPE one_minus_alpha_rev_value()
{
return (1.0/(1.0 - alpha));
}


__host__ void transfer_data_to_constant_memory(DATATYPE alpha,DATATYPE one_minus_alpha_rev)
{
CudaErrors(cudaMemcpyToSymbol(constant_alpha,&alpha,sizeof(DATATYPE)));
CudaErrors(cudaMemcpyToSymbol(constant_one_minus_alpha_rev,&one_minus_alpha_rev,sizeof(DATATYPE)));
}

__host__ void transfer_data_to_constant_memory(int lpat,int rsize,int norm)
{
CudaErrors(cudaMemcpyToSymbol(constant_lpat,&lpat,sizeof(int)));
CudaErrors(cudaMemcpyToSymbol(constant_rsize,&rsize,sizeof(int)));
CudaErrors(cudaMemcpyToSymbol(constant_norm,&norm,sizeof(int)));
}

__host__ double* Timer_Array(int timers)
{

double *timer=(double*)malloc(timers*sizeof(double));
if(timer==NULL)
{
exit(1);
}

for(unsigned int i=0; i<timers; i++)
{
timer[i]=0;
}

return timer;

}

__host__ void TimersFree(struct Timers *BT)
{
free(BT->init_timer);
free(BT->find_mdp_timer);
free(BT->merge_timer);
free(BT->clustering_timer);
free(BT->update_timer);
free(BT->pruning_timer);

free(BT);
}

__host__ char* current_path()
{
char cp[1024];

if(getcwd(cp,sizeof(cp))==NULL)
{
perror("getcwd() error");
return NULL;
}

return strdup(cp);
}

__host__ void change_directory(const char *path)
{
if (chdir(path) != 0) {
perror("chdir() error");
exit(1);
}
}

__host__ char* ConcenateString(const char *str1,const char *str2)
{

unsigned int length=strlen(str1)+strlen(str2)+1;

char *str3=(char*)malloc(length*sizeof(char));
if(str3==NULL)
{
exit(1);
}

strcpy(str3,str1);
strcat(str3,str2);

return str3;

}

__host__ BTree* alloc_item(struct BTree *item,int npat,int rsize,int lpat)
{

struct BTree *item1=(BTree*)malloc(npat*sizeof(BTree));
if(item1==NULL)
{
exit(1);
}

memcpy(item1,item,npat*sizeof(BTree));

for(unsigned int i=0 ;i<npat;i++)
{

item1[i].mean=(DATATYPE*)malloc(lpat*sizeof(DATATYPE));
if(item1[i].mean==NULL)
{
exit(1);
}
memcpy(item1[i].mean,item[i].mean,lpat*sizeof(DATATYPE));

item1[i].pats=(int*)calloc(1,nnpc*sizeof(int));
if(item1[i].pats==NULL)
{
exit(1);
}
memcpy(item1[i].pats,item[i].pats,nnpc*sizeof(int));

item1[i].rep=(DATATYPE**)malloc(rsize*sizeof(DATATYPE*));
if(item1[i].rep==NULL)
{
exit(1);
}

for(register int j=0;j<rsize;j++)
{
item1[i].rep[j]=(DATATYPE*)malloc(lpat*sizeof(DATATYPE));

if(item1[i].rep[j]==NULL)
{
exit(1);
}
}

register int maxrep=(item[i].size<=rsize)?item[i].size:rsize;
for(register int j=0;j<maxrep;j++)
{

memcpy(item1[i].rep[j],item[i].rep[j],lpat*sizeof(DATATYPE));
}

}

return item1;

}

__host__ void FreeItems(struct BTree *item,int npat,int rsize)
{
for(unsigned int i=0;i<npat;i++)
{
free(item[i].mean);
free(item[i].pats); 

for(register int j=0; j<rsize;j++)
{

free(item[i].rep[j]);

}
free(item[i].rep);
}

free(item); 
}

__host__ void show_time_results_for_each_case(const char *type,struct Timers *BT,int npat,int lpat,int csize)
{

char filename[1000];

if(BT == NULL)
{
exit(1);
}

sprintf(filename,"%s %d-cluster %d %d-d points Time Execution.txt",type,csize,npat,lpat);

FILE *file=fopen(filename,"w");

fprintf(file,"EXECUTION TIME REPORT\n");
fprintf(file,"INIT NNB TIME = %f\n", BT->init_timer[0]);
fprintf(file,"FIND MDP TIME = %f\n", BT->find_mdp_timer[0]);
fprintf(file,"CLUSTERING TIME = %f\n", BT->clustering_timer[0]);
fprintf(file,"\tMERGE TIME = %f\n", BT->merge_timer[0]);
fprintf(file,"\tUPDATE NNB TIME = %f\n", BT->update_timer[0]);
fprintf(file,"\tPRUNING TIME = %f\n", BT->pruning_timer[0]);
fprintf(file,"TOTAL TIME = INIT NNB + FIND MDP TIME + CLUSTERING = %f\n", BT->init_timer[0] + BT->find_mdp_timer[0] + BT->clustering_timer[0]);

fclose(file);

}

__host__ void print_rep_points(FILE *f, struct BTree *item, int cno, int rsize,int lpat)
{
register int i, j;

if (f == stdout)
printf("**** Cluster %d from CURE Rep Points ****\n", cno);
fprintf(f, "Cluster:%d\titem size:%d\tRepresantive Points:%d\n", cno, item->size, (item->size < rsize)? item->size: rsize);

fprintf(f, "C:\t");
for (j=0; j<lpat; ++j)
fprintf(f, "%f ", item->mean[j] - SHIFT_FACTOR);
fprintf(f, "\n");

for (i=0; i<item->size && i<rsize; ++i) {
fprintf(f, "R:\t");
for (j=0; j<lpat; ++j)
fprintf(f, "%f ", item->rep[i][j] - SHIFT_FACTOR);
fprintf(f, "\n");
}
}

__host__ void print_points(FILE *f, struct BTree *item, int cno, int lpat)
{
register int i;

fprintf(f,"Cluster Nodes:\t");
for (i=0; i<item->size; ++i) {
fprintf(f, "%d ", item->pats[i]);
}
fprintf(f, "\n\n");
}

__host__ void print_results(const char *type, struct BTree *item,int npat,int clusters,int rsize,int lpat)
{

char filename[1000];

sprintf(filename, "%s %d-Clusters %d %d-d points Results.txt",type,clusters,npat,lpat);

FILE *file=fopen(filename,"w");

fprintf(file, "NumberOfClusters:%d\n",clusters);

if(strcmp(type,"UnifiedMemory")==0 || strcmp(type,"PrefetchAsyncUnifiedMemory")==0 || strcmp(type,"PinnedMemoryCMH")==0 || strcmp(type,"PinnedMemoryCHA")==0 || strcmp(type,"MappedPinnedMemory")==0 || strcmp(type,"CudaCurePart")==0 || strcmp(type,"SerialCure")==0)
{
int cno = 0;
for (unsigned int i = 0; i < npat; i++) 
{
if (item[i].root == TRUE && item[i].pruned == FALSE) 
{
print_rep_points(file, &item[i], cno,rsize,lpat);
print_points(file, &item[i], cno, lpat);
++cno;
}
}	

}else if(strcmp(type,"AllDataGpu")==0)
{

for (unsigned int i = 0; i < clusters; i++) 
{

print_rep_points(file,&item[i],i,rsize,lpat);
print_points(file, &item[i],i, lpat);		
}
}	

fclose(file);	
}

__device__ double GPUdistance(DATATYPE *pat1, DATATYPE *pat2)
{

register int i;
double   dist = 0.0;

#pragma unroll
for (i = 0; i < constant_lpat; i++) {
double   diff = 0.0;


#if 0
#ifndef NO_DONTCARES
if (!IS_DC(pat1[i]) && !IS_DC(pat2[i]))

#endif
#endif
diff =pat1[i] - pat2[i];

switch (constant_norm) {
double   adiff;

case 2:
dist +=diff * diff;

break;
case 1:
dist += fabs(diff);

break;
case 0:
if ((adiff = fabs(diff)) > dist)
dist = adiff;

break;
default:
dist += pow(fabs(diff), (double) constant_norm);

break;
}
}

return dist;
}

__device__ double GPU_cure_distance(struct BTree *x,struct BTree *y)
{

register int i, j;
double min_dist = MAXFLOAT, dist;
int xmax, ymax;

if ((x->full == TRUE) || (y->full == TRUE)) {
return 1e99; 
}

if (x->size > constant_rsize)
xmax = constant_rsize;
else
xmax = x->size;

if (y->size > constant_rsize)
ymax = constant_rsize;
else
ymax = y->size;

#pragma unroll
for (i=0; i<xmax; ++i)
#pragma unroll
for (j=0; j<ymax; ++j)
if ((dist = GPUdistance(x->rep[i],y->rep[j])) < min_dist)
min_dist = dist;

return min_dist;
}


__global__ void BTree_transfer(struct BTree *item,DATATYPE *rep,DATATYPE *mean,int rsize,int lpat)
{
unsigned int tidx=threadIdx.x,tidy=threadIdx.y;
unsigned int bidx=blockIdx.x,bidy=blockIdx.y;
unsigned int blocksizex=blockDim.x,blocksizey=blockDim.y;
unsigned int i=tidx+bidx*blocksizex,j=tidy+bidy*blocksizey;

if(i>=rsize || j>=lpat) return;

rep[i*lpat+j]=item->rep[i][j];

if(i==0)
{
mean[j]=item->mean[j];
}


}

__global__ void pats_transfer(int **pats,int *pats1)
{
unsigned int bid=blockIdx.x;
unsigned int blocksize=blockDim.x;
unsigned int tid=threadIdx.x;

unsigned int i = tid +bid*blocksize;

pats1[i]=pats[0][i];

}

__global__ void GPUroot(double *dist,double *gmin_dist)
{

if (dist == 0)
{
*dist=0;
return;
}

switch (constant_norm) {
case 2:
*dist=sqrt(gmin_dist[0]);
return;
case 1:
case 0:
*dist=gmin_dist[0];
return;
default:
*dist=pow(gmin_dist[0], 1 / (double) constant_norm);
return;
}

}

__global__ void GPUmerge(struct BTree *item,struct BTree *merge_tmp_item,int **pats,int pair1,int pair2,double *min_dist,int size_limit)
{

double maxDist;
DATATYPE *repPoint;
double dist;
static	int point_used[64];
unsigned int newsize=item[pair1].size+item[pair2].size;
DATATYPE r1 = (DATATYPE)item[pair1].size/newsize;
DATATYPE r2 = (DATATYPE)item[pair2].size/newsize;

int pair1_npoints = (constant_rsize <= item[pair1].size)? constant_rsize: item[pair1].size;
int pair2_npoints = (constant_rsize <= item[pair2].size)? constant_rsize: item[pair2].size;

if ((item[pair1].size > constant_rsize)&&(constant_alpha < 1.0)) {
for (unsigned int j=0; j<constant_rsize; ++j)
for (unsigned int k=0; k<constant_lpat; ++k)
item[pair1].rep[j][k] = constant_one_minus_alpha_rev*(item[pair1].rep[j][k] - constant_alpha*item[pair1].mean[k]);

}

if ((item[pair2].size > constant_rsize)&&(constant_alpha < 1.0)) {
for (unsigned int j=0; j<constant_rsize; ++j)
for (unsigned int k=0; k<constant_lpat; ++k)
item[pair2].rep[j][k] = constant_one_minus_alpha_rev*(item[pair2].rep[j][k] - constant_alpha*item[pair2].mean[k]);

}

merge_tmp_item->size = item[pair1].size;
merge_tmp_item->distance = item[pair1].distance;

unsigned int k = item[pair1].size;
unsigned int l = item[pair2].size;


for (unsigned int m = 0; m < l; m++)
{
pats[pair1][m+k] = pats[pair2][m];
}

for (unsigned int i=0; i<constant_lpat; ++i)
{ 
item[pair1].mean[i] = r1*item[pair1].mean[i]+ r2*item[pair2].mean[i];
}

item[pair1].size = newsize;
if (newsize > size_limit) item[pair1].full = TRUE;
item[pair1].distance = *min_dist;

item[pair2].root = FALSE;

if (newsize <= constant_rsize) {

for (unsigned int i=pair1_npoints, j = 0; j < pair2_npoints; ++i, ++j) 
{

item[pair1].rep[i] = item[pair2].rep[j];

}

}else
{


for (unsigned int i=0; i < pair1_npoints; ++i) 
{

merge_tmp_item->rep[i]=item[pair1].rep[i]; 
}


for (unsigned int i=pair1_npoints, j = 0; j < pair2_npoints; ++i, ++j) 
{

merge_tmp_item->rep[i]=item[pair2].rep[j];
}


for (unsigned int i = 0; i < (pair1_npoints+pair2_npoints); i++) point_used[i] = 0;

for (unsigned int rcount=0; rcount < constant_rsize; ++rcount) 
{

int point_id = -1;
maxDist = -0.1;

for (unsigned int i = 0; i < (pair1_npoints+pair2_npoints); i++) 
{

if (point_used[i]) continue;

if(rcount==0)
{

dist = GPUdistance(merge_tmp_item->rep[i], item[pair1].mean);


if (dist > maxDist) 
{
maxDist = dist;
repPoint = merge_tmp_item->rep[i];
point_id = i;

}


}else
{
dist = GPUdistance(merge_tmp_item->rep[i], item[pair1].rep[rcount-1]);

if (dist > maxDist) 
{
maxDist = dist;
repPoint = merge_tmp_item->rep[i];
point_id = i;

}

}

}

point_used[point_id] = 1;

item[pair1].rep[rcount]=repPoint;

}

for (unsigned int j=0; j< constant_rsize; ++j)
for (unsigned int k=0; k<constant_lpat; ++k)
item[pair1].rep[j][k] += (constant_alpha*(item[pair1].mean[k]-
item[pair1].rep[j][k]));
																																																														  
}

}

__global__ void GPU_initialization_phase1(struct BTree *item,struct nnb_info *GPU_nnb,int npat)
{

unsigned int tid=threadIdx.x;
unsigned int bid=blockIdx.x;
unsigned int blocksize=blockDim.x;
unsigned int i=tid+bid*blocksize;

if(i>=npat) return;

int index=NONE;
double dist=0.0;

if(item[i].root == TRUE && item[i].full == FALSE)
{

for(unsigned int j=0; j<i;j++)
{

if(item[j].root== TRUE && item[j].pruned== FALSE && item[j].full== FALSE)
{

double distance=GPU_cure_distance(&item[i],&item[j]);

if(index==NONE || distance<dist)
{
index=j;
dist=distance;
}

}

}


}

GPU_nnb[i].index=index;
GPU_nnb[i].dist=dist;
}

__global__ void NO_NONE_MDP(struct BTree *item,struct nnb_info *GPU_nnb,int *GPU_gpair1,double *GPU_gmin_dist,int *counter,int npat)
{

unsigned int tid=threadIdx.x;
unsigned int bid=blockIdx.x;
unsigned int blocksize=blockDim.x;
unsigned int i=tid+bid*blocksize;

if(i>=npat) return;

double dist1=GPU_nnb[i].dist;

if (item[i].root == TRUE && item[i].pruned == FALSE && item[i].full == FALSE && GPU_nnb[i].index != NONE )
{
unsigned int index=atomicAdd(counter,1);
GPU_gpair1[index]=i;
GPU_gmin_dist[index]=dist1;       
}

}

__global__ void GPU_find_minimum_distance_pair_per_block(int *GPU_lpair1,double *GPU_lmin_dist,int *GPU_gpair1,double *GPU_gmin_dist,int npat)
{

__shared__ int shared_lpair1[MAXBLOCKSIZE];
__shared__ double shared_lmin_dist[MAXBLOCKSIZE];

unsigned int tid=threadIdx.x;
unsigned int bid=blockIdx.x;
unsigned int blocksize=blockDim.x;
unsigned int j=tid+bid*blocksize;

if(j>=npat) return;

shared_lpair1[tid]=GPU_lpair1[j];
shared_lmin_dist[tid]=GPU_lmin_dist[j];

__syncthreads();

for(int i=blocksize/2; i>0; i>>=1)
{

if(tid<i && ((j+i)<npat))
{
if(((shared_lmin_dist[tid]>shared_lmin_dist[tid+i]) || ((shared_lmin_dist[tid] == shared_lmin_dist[tid+i])&&(shared_lpair1[tid]>shared_lpair1[tid+i]))))
{

shared_lmin_dist[tid]=shared_lmin_dist[tid+i];
shared_lpair1[tid]=shared_lpair1[tid+i];

}

}

__syncthreads();		 
}


if(tid==0)
{

GPU_gpair1[bid] = shared_lpair1[0];
GPU_gmin_dist[bid] = shared_lmin_dist[0];     
}
__syncthreads();     
}

__global__ void GPU_find_minimum_distance_pair(int *index ,double *distances,int npat)
{

__shared__ int shared_index[MAXBLOCKSIZE];
__shared__ double shared_distances[MAXBLOCKSIZE];

unsigned int tid=threadIdx.x;
unsigned int bid=blockIdx.x;
unsigned int blocksize=blockDim.x;
unsigned int j=tid+bid*blocksize;

if(j>=npat) return;

shared_index[tid]=index[j];
shared_distances[tid]=distances[j];

__syncthreads();

for(int i=blocksize/2; i>0; i>>=1)
{
if(tid<i &&((j+i)<npat)&&((shared_distances[tid]>shared_distances[tid+i]) || ( (shared_distances[tid] == shared_distances[tid+i])&&(shared_index[tid]>shared_index[tid+i]))))
{
shared_distances[tid]=shared_distances[tid+i];
shared_index[tid]=shared_index[tid+i];

}

__syncthreads();
}

if(tid==0)
{
index[bid]=shared_index[0];
distances[bid]=shared_distances[0];
}

__syncthreads();

}

__global__ void GPU_update_nnb_kernel(struct BTree *item,struct nnb_info *GPU_nnb,int npat,int *GPU_gpair1,double *GPU_gmin_dist)
{

unsigned int tid=threadIdx.x;
unsigned int bid=blockIdx.x;
unsigned int blocksize=blockDim.x;
unsigned int i=tid+bid*blocksize;

if(i>=npat) return;

int lpair1;
double lmin_dist;

lpair1=NONE;
lmin_dist=-0.1;

if((GPU_nnb[i].index != NONE && item[i].root== TRUE && item[i].pruned== FALSE && item[i].full== FALSE))
{

lpair1=i;
lmin_dist=GPU_nnb[i].dist;

}


GPU_gpair1[i]=lpair1;
GPU_gmin_dist[i]=lmin_dist;

}

__global__ void GPU_update_nnb_kernel1(struct BTree *item,struct nnb_info *GPU_nnb,int npat,int pair1,int pair2,int *GPU_gpair1,double *GPU_gmin_dist,int offset)
{

unsigned int tid=threadIdx.x;
unsigned int bid=blockIdx.x;
unsigned int blocksize=blockDim.x;
unsigned int i=offset+tid+bid*blocksize;


if(i>=npat) return;


int lpair1=NONE;
double lmin_dist=-0.1;

if ((GPU_nnb[i].index != NONE && item[i].root== TRUE && item[i].pruned== FALSE && item[i].full== FALSE) && (GPU_nnb[i].index !=pair1 && GPU_nnb[i].index != pair2)) {

lpair1 = i;
lmin_dist = GPU_nnb[i].dist;   


}

GPU_gpair1[i]=lpair1;
GPU_gmin_dist[i]=lmin_dist;

}


__global__ void GPU_update_nnb_kernel2(struct BTree *item,struct nnb_info *GPU_nnb,int npat,int pair1,int pair2,int *GPU_gpair1,double *GPU_gmin_dist,int offset)
{

__shared__ BTree shared_item[1];

unsigned int tid=threadIdx.x;
unsigned int bid=blockIdx.x;
unsigned int blocksize=blockDim.x;
unsigned int i=offset+tid+bid*blocksize;

if(i>=npat) return;

shared_item[0]=item[pair1];

__syncthreads();

int lpair1=NONE;
double lmin_dist=-0.1;

if ((GPU_nnb[i].index != NONE && item[i].root== TRUE && item[i].pruned== FALSE && item[i].full== FALSE) && ((GPU_nnb[i].index !=pair1) && (GPU_nnb[i].index != pair2)))
{



double distance=GPU_cure_distance(&shared_item[0],&item[i]); 

if(distance<GPU_nnb[i].dist)
{
GPU_nnb[i].index=pair1;
GPU_nnb[i].dist=distance;
}

lpair1 = i;
lmin_dist = GPU_nnb[i].dist;            


}


GPU_gpair1[i]=lpair1;
GPU_gmin_dist[i]=lmin_dist;

}


__global__ void GPU_update_nnb_kernel3(struct BTree *item,struct nnb_info *GPU_nnb,int npat,int pair1,int pair2,int offset,int *counter,int*indices)
{

__shared__ int shared_root[MAXBLOCKSIZE];
__shared__ int shared_pruned[MAXBLOCKSIZE];
__shared__ int shared_full[MAXBLOCKSIZE];

__shared__ nnb_info shared_nnb[MAXBLOCKSIZE];

unsigned int tid=threadIdx.x;
unsigned int bid=blockIdx.x;
unsigned int blocksize=blockDim.x;
unsigned int i=offset+tid+bid*blocksize;

if(i>=npat) return;

shared_nnb[tid]=GPU_nnb[i];

shared_root[tid]=item[i].root;
shared_pruned[tid]=item[i].pruned;
shared_full[tid]=item[i].full;

__syncthreads();

if ((shared_nnb[tid].index != NONE && shared_root[tid]== TRUE && shared_pruned[tid]== FALSE && shared_full[tid]== FALSE) && ((shared_nnb[tid].index ==pair1) || (shared_nnb[tid].index == pair2)))
{

unsigned int index=atomicAdd(counter,1);
indices[index]=i;

}

}


__global__ void GPU_update_kernel(struct BTree *item,int npat,int *indices,double *distances,int *counter)
{

__shared__ int indices1[MAXBLOCKSIZE];
__shared__ double distances1[MAXBLOCKSIZE];

unsigned int tid=threadIdx.x;
unsigned int bid=blockIdx.x;
unsigned int blocksize=blockDim.x;
unsigned int j=tid+bid*blocksize;

if(j>=npat) return;

int index=NONE;
double dist=-0.1;


if(item[npat].root == TRUE && item[npat].full == FALSE)
{

if(item[j].root== TRUE && item[j].pruned== FALSE && item[j].full== FALSE)
{

dist=GPU_cure_distance(&item[npat],&item[j]);

index=j;

}

}


indices1[tid]=index;
distances1[tid]=dist;


__syncthreads();

for(int i=blocksize/2; i>0; i>>=1)
{

if(tid<i && ((j+i)<npat))
{
if((indices1[tid]!=NONE && indices1[tid+i] != NONE) && ((distances1[tid]>distances1[tid+i]) || ((distances1[tid] == distances1[tid+i])&&(indices1[tid]>indices1[tid+i]))))
{

distances1[tid]=distances1[tid+i];
indices1[tid]=indices1[tid+i];

}else if(indices1[tid]==NONE && indices1[tid+i]!=NONE)
{
distances1[tid]=distances1[tid+i];
indices1[tid]=indices1[tid+i];

}

}

__syncthreads();
}


if(tid==0)
{
if(indices1[0] != NONE)
{
unsigned int index=atomicAdd(counter,1);

indices[index] = indices1[0];
distances[index] = distances1[0];
}
}
__syncthreads();

}

__global__ void GPU_update_kernel1(int npat,int *indices,double *distances)
{

__shared__ int indices1[MAXBLOCKSIZE];
__shared__ double distances1[MAXBLOCKSIZE];

unsigned int tid=threadIdx.x;
unsigned int bid=blockIdx.x;
unsigned int blocksize=blockDim.x;
unsigned int j=tid+bid*blocksize;

if(j>=npat) return;

indices1[tid]=indices[j];
distances1[tid]=distances[j];

__syncthreads();

for(int i=blocksize/2; i>0; i>>=1)
{

if(tid<i && ((j+i)<npat))
{
if((indices1[tid]!=NONE && indices1[tid+i] != NONE) && ((distances1[tid]>distances1[tid+i]) || ((distances1[tid] == distances1[tid+i])&&(indices1[tid]>indices1[tid+i]))))
{

distances1[tid]=distances1[tid+i];
indices1[tid]=indices1[tid+i];

}else if(indices1[tid]==NONE && indices1[tid+i]!=NONE)
{
distances1[tid]=distances1[tid+i];
indices1[tid]=indices1[tid+i];

}

}

__syncthreads();
}

if(tid==0)
{
indices[bid]=indices1[tid];
distances[bid]=distances1[tid];
}

__syncthreads();

}

__global__ void GPU_update_kernel2(struct BTree *item,struct nnb_info *GPU_nnb,int node,int *indices,double *distances,int *lpair1,double *lmin_dist)
{

int lpair2=NONE;
double lmin_dist2=-0.1;

GPU_nnb[node].index=indices[0];

if(GPU_nnb[node].index>=0)
{
GPU_nnb[node].dist=distances[0];
} 


if ((GPU_nnb[node].index != NONE)) 
{
lpair2 = node;
lmin_dist2 = GPU_nnb[node].dist;           
}

__syncthreads();

lpair1[node]=lpair2;
lmin_dist[node]=lmin_dist2;

}


__global__ void GPU_update_kernel3(int *lpair1,double *lmin_dist,int *indices,double *distances,int npat,int *counter)
{
__shared__ int shared_lpair1[MAXBLOCKSIZE];
__shared__ double shared_lmin_dist[MAXBLOCKSIZE];

unsigned int tid=threadIdx.x;
unsigned int bid=blockIdx.x;
unsigned int blocksize=blockDim.x;
unsigned int j=tid+bid*blocksize;

if(j>=npat) return;

shared_lpair1[tid]=lpair1[j];
shared_lmin_dist[tid]=lmin_dist[j];

__syncthreads();

for(int i=blocksize/2; i>0; i>>=1)
{

if(tid<i && ((j+i)<npat))
{
if((shared_lpair1[tid]!=NONE && shared_lpair1[tid+i] != NONE) && ((shared_lmin_dist[tid]>shared_lmin_dist[tid+i]) || ((shared_lmin_dist[tid] == shared_lmin_dist[tid+i])&&(shared_lpair1[tid]>shared_lpair1[tid+i]))))
{

shared_lmin_dist[tid]=shared_lmin_dist[tid+i];
shared_lpair1[tid]=shared_lpair1[tid+i];

}else if(shared_lpair1[tid]==NONE && shared_lpair1[tid+i]!=NONE)
{
shared_lmin_dist[tid]=shared_lmin_dist[tid+i];
shared_lpair1[tid]=shared_lpair1[tid+i];

}

}

__syncthreads();
}


if(tid==0)
{
if(shared_lpair1[0] != NONE)
{
unsigned int index=atomicAdd(counter,1);

indices[index] = shared_lpair1[0];
distances[index] = shared_lmin_dist[0];
}                                         
}

__syncthreads(); 

}

__global__ void GPU_update_kernel4(int *indices,double *distances,int npat,int counter)
{

unsigned int tid=threadIdx.x;
unsigned int bid=blockIdx.x;
unsigned int blocksize=blockDim.x;
unsigned int i=tid+bid*blocksize;

if(i>=npat) return;

if(counter==0)
{
indices[i]=NONE;
distances[i]=-0.1;
}

}


__global__ void GPU_first_pruning_kernel(struct BTree *item,int npat,int*nodes,int *pruned_nodes)
{

__shared__ int shared_pruned[MAXBLOCKSIZE];

__shared__ int shared_nodes;

unsigned int tid=threadIdx.x;
unsigned int bid=blockIdx.x;
unsigned int blocksize=blockDim.x;
unsigned int i=tid+bid*blocksize;

if(i>=npat) return;

shared_pruned[tid]=item[i].pruned;

if(tid==0)
{
shared_nodes=0;
}

__syncthreads();

if(item[i].root== TRUE && item[i].size< FirstCut)
{
atomicAdd(&shared_nodes,1);
shared_pruned[tid]=TRUE;
}

__syncthreads();


if(tid==0)
{

atomicSub(nodes,shared_nodes);
atomicAdd(pruned_nodes,shared_nodes);

}

item[i].pruned=shared_pruned[tid];

__syncthreads();

}

__global__ void GPU_second_pruning_kernel(struct BTree *item,int npat,int *nodes,int *pruned_nodes)
{

__shared__ int  shared_pruned[MAXBLOCKSIZE];

__shared__ int shared_nodes;

unsigned int tid=threadIdx.x;
unsigned int bid=blockIdx.x;
unsigned int blocksize=blockDim.x;
unsigned int i=tid+bid*blocksize;

if(i>=npat) return;

shared_pruned[tid]=item[i].pruned;

if(tid==0)
{
shared_nodes=0;
}

__syncthreads();

if(item[i].root == TRUE && item[i].full == FALSE && item[i].pruned == FALSE && item[i].size < SecondCut)
{
shared_pruned[tid]=TRUE;
atomicAdd(&shared_nodes,1);

}

__syncthreads();

if(tid==0)
{
atomicSub(nodes,shared_nodes);
atomicAdd(pruned_nodes,shared_nodes);
}

item[i].pruned=shared_pruned[tid];

__syncthreads();  

}


__global__ void GPU_pruning_kernel1(struct BTree *item,struct nnb_info *GPU_nnb,int npat,int *counter,int *indices)
{

unsigned int tid=threadIdx.x;
unsigned int bid=blockIdx.x;
unsigned int blocksize=blockDim.x;
unsigned int i=tid+bid*blocksize;

if(i>=npat) return;

if((GPU_nnb[i].index !=NONE) && (item[i].root ==  TRUE && item[GPU_nnb[i].index].pruned == TRUE && item[i].full == FALSE))
{

unsigned int index=atomicAdd(counter,1);
indices[index]=i;

}


}

__global__ void GPU_pruning_kernel2(struct BTree *item,struct nnb_info *GPUnnb,int *indices,int npat,int *lpair1,double *lmin_dist1)
{

struct nnb_info shared_nnb[MAXBLOCKSIZE];

unsigned int tid=threadIdx.x;
unsigned int bid=blockIdx.x;
unsigned int blocksize=blockDim.x;
unsigned int i=tid+bid*blocksize;

if(i >= npat) return;

shared_nnb[tid]=GPUnnb[indices[i]];

int index=NONE,lpair=NONE;
double dist=0.0,lmin_dist=-0.1;

if(item[indices[i]].root == TRUE && item[indices[i]].full == FALSE)
{
for(unsigned int j=0; j<indices[i];j++)
{

if(item[j].root== TRUE && item[j].pruned== FALSE && item[j].full== FALSE)
{
double distance=GPU_cure_distance(&item[indices[i]],&item[j]);

if(index==NONE || distance<dist)
{
index=j;
dist=distance;
}

}

}

} 

shared_nnb[tid].index=index;

if(shared_nnb[tid].index>=0)
{
shared_nnb[tid].dist=dist;
}

if (shared_nnb[tid].index != NONE) {

lpair = indices[i];
lmin_dist = shared_nnb[tid].dist;   
}                 

lpair1[indices[i]]=lpair;
lmin_dist1[indices[i]]=lmin_dist;

GPUnnb[indices[i]].index=shared_nnb[tid].index;
GPUnnb[indices[i]].dist=shared_nnb[tid].dist;

}