D ?= 2
KNN ?= 10
KNC ?= 2
QUE = 10

SERIALFOLDER   :=  SerialCure
PARTFOLDER   :=  CudaCurePart
SINGLECOMMONFOLDER   :=  SingleCudaCure
VISUALIZATIONDATAFOLDER2D  :=  VisualizationData/SavePointsFigures/2DPoints
MemoryTypes  :=  SERIALCURE  PCUDACURE  ALLCUDACURE  UMCUDACURE  PAUMCUDACURE  CMHCUDACURE  CHACUDACURE  MCUDACURE

CFLAGS  = -DPROBDIM=$D -DNNBS=$(KNN) -DQUERYELEMS=$(QUE) -DNNCS=$(KNC)

CC = gcc
CFLAGS += -g -ggdb
LDFLAGS += -lm 

NVCC = nvcc 
CUDAFLAGS = -DPROBDIM=$D -DNNBS=$(KNN) -DQUERYELEMS=$(QUE) -DNNCS=$(KNC)
CUDAFLAGS += -g
##########################################
# Compilation Rules
#

PROG_OMP	= cluster_omp.exe
HDRS_OMP	= alloc.h cluster.h help.h CudaCheckError/cudacheckerror.cuh CudaTimer/cudatimer.cuh CudaKernelsHostDeviceFunctions/KernelsHostDeviceFunctions.cuh Timers/Timers.h     
SRCS_OMP	= alloc.c cluster_omp.c    
OBJS_OMP	= alloc.o cluster_omp.o    

CUOBJS_OMP      = alloc.o cluster_omp.o cudacheckerror.o cudatimer.o KernelsHostDeviceFunctions.o AllDataGpu.o SingleCudaCure.o CudaCurePart.o UnifiedMemory.o PrefetchAsyncUnifiedMemory.o PinnedMemoryCMH.o PinnedMemoryCHA.o MappedPinnedMemory.o CudaCheckers.o    

all: $(PROG_OMP)

$(PROG_OMP):	$(CUOBJS_OMP)

	$(NVCC) $(CUDAFLAGS) -o $(PROG_OMP) $(CUOBJS_OMP)
	rm -f *.o

alloc.o: alloc.c $(HDRS_OMP)
cluster_omp.o: cluster_omp.c $(HDRS_OMP)
	$(NVCC) -c cluster_omp.c
assign.o: assign.c
    
cudacheckerror.o: CudaCheckError/cudacheckerror.cu 
	$(NVCC) -c CudaCheckError/cudacheckerror.cu

cudatimer.o: CudaTimer/cudatimer.cu
	$(NVCC) -c CudaTimer/cudatimer.cu
  
KernelsHostDeviceFunctions.o: CudaKernelsHostDeviceFunctions/KernelsHostDeviceFunctions.cu  
	$(NVCC) -c CudaKernelsHostDeviceFunctions/KernelsHostDeviceFunctions.cu  
       
CudaCurePart.o: CudaCurePart/CudaCurePart.cu
	$(NVCC) -c CudaCurePart/CudaCurePart.cu
  
AllDataGpu.o: SingleCudaCure/AllDataGpu/AllDataGpu.cu
	$(NVCC) -c SingleCudaCure/AllDataGpu/AllDataGpu.cu
   
UnifiedMemory.o: SingleCudaCure/UnifiedMemory/UnifiedMemory.cu
	$(NVCC) -c SingleCudaCure/UnifiedMemory/UnifiedMemory.cu
  
PrefetchAsyncUnifiedMemory.o: SingleCudaCure/PrefetchAsyncUnifiedMemory/PrefetchAsyncUnifiedMemory.cu
	$(NVCC) -c SingleCudaCure/PrefetchAsyncUnifiedMemory/PrefetchAsyncUnifiedMemory.cu  
   
PinnedMemoryCMH.o: SingleCudaCure/PinnedMemoryCMH/PinnedMemoryCMH.cu
	$(NVCC) -c SingleCudaCure/PinnedMemoryCMH/PinnedMemoryCMH.cu 
  
PinnedMemoryCHA.o: SingleCudaCure/PinnedMemoryCHA/PinnedMemoryCHA.cu
	$(NVCC) -c SingleCudaCure/PinnedMemoryCHA/PinnedMemoryCHA.cu 
   
MappedPinnedMemory.o: SingleCudaCure/MappedPinnedMemory/MappedPinnedMemory.cu
	$(NVCC) -c SingleCudaCure/MappedPinnedMemory/MappedPinnedMemory.cu
   
CudaCheckers.o: CudaCheckers/CudaCheckers.cu
	$(NVCC) -c CudaCheckers/CudaCheckers.cu 

SingleCudaCure.o: SingleCudaCure/SingleCudaCure.cu
	$(NVCC) -c SingleCudaCure/SingleCudaCure.cu
            
# Makefile rules

.c.o:
	${CC} ${CFLAGS}  -c $*.c
 
.cu.o:
	$(NVCC) $(CUDAFLAGS) -c $*.cu

debug:
	cuda-gdb 

clean:
	rm -f *.o
  
cleanall:
	rm -f *exe *.o *.out assign.*.txt  \
  rm -f  $(SERIALFOLDER)/*.txt  \
  rm -f  $(PARTFOLDER)/*.txt  \
  rm -f  $(SINGLECOMMONFOLDER)/*/*.txt  \
  rm -f  $(VISUALIZATIONDATAFOLDER2D)/*.png  \
                     
clear:
	rm -f *.out

# clear results 
cleanR:
	 rm -f  $(SERIALFOLDER)/*Results.txt  \
    rm -f  $(PARTFOLDER)/*Results.txt  \
 rm -f  $(SINGLECOMMONFOLDER)/*/*Results.txt  \

# clean time execution
cleanT:
	 rm -f  $(SERIALFOLDER)/*Execution.txt  \
    rm -f  $(PARTFOLDER)/*Execution.txt  \
 rm -f  $(SINGLECOMMONFOLDER)/*/*Execution.txt  \
 
#clean time execution and time results 
cleanTR:
	 rm -f  $(SERIALFOLDER)/*.txt  \
    rm -f  $(PARTFOLDER)/*.txt  \
 rm -f  $(SINGLECOMMONFOLDER)/*/*.txt  \ 

#clean images 
cleanI:
	rm -f  $(VISUALIZATIONDATAFOLDER2D)/*.png  \

 
# Run and create 4 clusters for 200 2d points 
CheckRun:
	for  Mem  in  $(MemoryTypes)  ;  do  \
   ./cluster_omp.exe -k 4 -p 200 -d 2 -x  ex2.dat -T $$Mem;  \
   echo  "\n\n";  \
 done;  \
 echo  "\n";  \
 echo  "Begin Clusters Visualization";  \
 rm -f  $(SERIALFOLDER)/*Execution.txt  \
 rm -f  $(PARTFOLDER)/*Execution.txt  \
 rm -f  $(SINGLECOMMONFOLDER)/*/*Execution.txt  \
 rm -f  $(VISUALIZATIONDATAFOLDER2D)/*.png  \
 echo  "\n";  \
 python3  VisualizationData/VisualizationData.py;  \
 echo  "End Clusters Visualization";  \
  
# Run for different memory types and number of 20 d points 
TimeRun:
	for  Mem  in  $(MemoryTypes)  ;  do  \
   ./cluster_omp.exe -k 5 -p 500 -d 20 -x  ex2.dat -T $$Mem;  \
   echo  "\n";  \
   ./cluster_omp.exe -k 10 -p 1000 -d 20 -x  ex2.dat -T $$Mem;  \
   echo  "\n";  \
   ./cluster_omp.exe -k 15 -p 1500 -d 20 -x  ex2.dat -T $$Mem;  \
   echo  "\n";  \
   ./cluster_omp.exe -k 30 -p 3000 -d 20 -x  ex2.dat -T $$Mem;  \
   echo  "\n";  \
   ./cluster_omp.exe -k 60 -p 6000 -d 20 -x  ex2.dat -T $$Mem;  \
   echo  "\n";  \
   ./cluster_omp.exe -k 120 -p 12000 -d 20 -x  ex2.dat -T $$Mem;  \
   echo  "\n";  \
   ./cluster_omp.exe -k 240 -p 24000 -d 20 -x  ex2.dat -T $$Mem;  \
   echo  "\n";  \
   ./cluster_omp.exe -k 480 -p 48000 -d 20 -x  ex2.dat -T $$Mem;  \
   echo  "\n";  \
 done;  \
 rm -f  $(SERIALFOLDER)/*Results.txt  \
 rm -f  $(PARTFOLDER)/*Results.txt  \
 rm -f  $(SINGLECOMMONFOLDER)/*/*Results.txt  \
 