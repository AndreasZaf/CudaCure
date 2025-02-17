import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import re


# Define file paths for different variations of the CURE algorithm

FilePaths=[
 "SerialCure/", # Results for the serial version of CURE
 "SingleCudaCure/AllDataGpu/", # CURE execution with all data on the GPU
 "SingleCudaCure/UnifiedMemory/", # Use of Unified Memory in the GPU
 "SingleCudaCure/PrefetchAsyncUnifiedMemory/", # Unified memory with prefetc async
 "SingleCudaCure/PinnedMemoryCMH/", # Use of pinned memory with cudaMallocHost
 "SingleCudaCure/PinnedMemoryCHA/", # Another variant of pinned memory with cudaHostAlloc
 "SingleCudaCure/MappedPinnedMemory/", # Mapped pinned memory
 "CudaCurePart/"
];

# Function to combine the 2D results and save the image

def Combine() :
  
  fig.tight_layout() # Adjust layout for better spacing
  save_path = os.path.join("VisualizationData/SavePointsFigures/2DPoints","2DCombine"+".png") # Set save path
  fig.savefig(save_path,dpi=300,bbox_inches="tight"); # Save image with high resolution
   
# Function to plot cluster results
   
def printcluster(Dictionary) :
   
   Clusters=Dictionary["NumberOfClusters"]; # Retrieve number of clusters created
   
   if Clusters <= 4 :   # If there are less than 4 clusters, proceed with visualizatio
      
    data=[[] for _ in range(Clusters)]; # Create lists to store data points for each cluster
    label=[];
    
    
    # Collect points for each cluster
   
    for key in range(Clusters) :
        key=str(key);
        if key in Dictionary :
            label.append(int(key));
            data[int(key)].append(Dictionary[key]["Centre Points"]); # Center point of the cluster
            data[int(key)].extend(Dictionary[key]["Represantive Points"]); # Representative points for the cluster
   
    Dimension=len(data[0][0]); # Check if it's 2D or 3D data
    
    if Dimension < 3 :  # Visualization for 2D data
    
     title=Dictionary["Type"];  
     cmap = plt.get_cmap("viridis"); # Use color map for different clusters 
     Colors=cmap(np.linspace(0,1,Clusters)); # Assign colors to each cluster 
   
     dotsize=100-(Clusters-1)*10; # Adjust dot size depending on number of clusters
                            
     label=np.array(label);
     data=np.array(data);
     
     plt.figure(title,figsize=(8,6));  # Create a new figure
      
     for clusters in np.unique(label) :
       clusterpoints=data[clusters==label];
       Points=len(clusterpoints[0]);
       if "SerialCure" in title :
       # Plot the cluster centers and points
         if clusters ==0 :
           plt.scatter(clusterpoints[0][0][0],clusterpoints[0][0][1],color="RED",s=dotsize,label=f"Centre Points");
           ax1.scatter(clusterpoints[0][0][0],clusterpoints[0][0][1],color="RED",s=dotsize,label=f"Centre Points");
         else :
           plt.scatter(clusterpoints[0][0][0],clusterpoints[0][0][1],color="RED",s=dotsize);   
           ax1.scatter(clusterpoints[0][0][0],clusterpoints[0][0][1],color="RED",s=dotsize);
         for i in range(1,Points) :
          if i == 1 :
            plt.scatter(clusterpoints[0][i][0],clusterpoints[0][i][1],color=Colors[clusters],s=dotsize,label=f"Cluster {clusters}");
            ax1.scatter(clusterpoints[0][i][0],clusterpoints[0][i][1],color=Colors[clusters],s=dotsize,label=f"Cluster {clusters}");
          else : 
            plt.scatter(clusterpoints[0][i][0],clusterpoints[0][i][1],color=Colors[clusters],s=dotsize); 
            ax1.scatter(clusterpoints[0][i][0],clusterpoints[0][i][1],color=Colors[clusters],s=dotsize); 
       elif "CudaCurePart" in title :
       # Visualization for CUDA CURE
         if clusters ==0 :
           plt.scatter(clusterpoints[0][0][0],clusterpoints[0][0][1],marker="x",color="RED",s=dotsize,label=f"Centre Points");
           ax2.scatter(clusterpoints[0][0][0],clusterpoints[0][0][1],marker="x",color="RED",s=dotsize,label=f"Centre Points");
         else :
           plt.scatter(clusterpoints[0][0][0],clusterpoints[0][0][1],marker="x",color="RED",s=dotsize);   
           ax2.scatter(clusterpoints[0][0][0],clusterpoints[0][0][1],marker="x",color="RED",s=dotsize);
         for i in range(1,Points) :
          if i == 1 :
            plt.scatter(clusterpoints[0][i][0],clusterpoints[0][i][1],marker="x",color=Colors[clusters],s=dotsize,label=f"Cluster {clusters}");
            ax2.scatter(clusterpoints[0][i][0],clusterpoints[0][i][1],marker="x",color=Colors[clusters],s=dotsize,label=f"Cluster {clusters}");
          else : 
            plt.scatter(clusterpoints[0][i][0],clusterpoints[0][i][1],marker="x",color=Colors[clusters],s=dotsize);
            ax2.scatter(clusterpoints[0][i][0],clusterpoints[0][i][1],marker="x",color=Colors[clusters],s=dotsize);
                 
     # Add labels and titles to the plot
                              
     plt.xlabel("X");
     plt.ylabel("Y");
     plt.title(title);
     plt.legend(fontsize="small", loc="best", markerscale=0.5)      
     save_path = os.path.join("VisualizationData/SavePointsFigures/2DPoints",title+".png")
     plt.savefig(save_path,dpi=300,bbox_inches="tight"); 
                      
     if "SerialCure" in title :
      # Adjust and display subplots for SerialCure
      ax1.set_xlabel("X");
      ax1.set_ylabel("Y");
      ax1.set_title(title);
      ax1.legend(fontsize="small", loc="best", markerscale=0.5) 
      ax1.grid(False);
     elif "CudaCurePart" in title :
     # Adjust and display subplots for CudaCurePar
      ax2.set_xlabel("X");
      ax2.set_ylabel("Y");
      ax2.set_title(title);
      ax2.legend(fontsize="small", loc="best", markerscale=0.5)
      ax2.grid(False);
      
# Function to check if two cluster result dictionaries are identical      
                          
def SameResults(SC,CCP) : 
 
 SC1=SC.copy();
 CCP1=CCP.copy();
 
 SC1.pop("Type",None); # Remove the "Type" key for comparison
 CCP1.pop("Type",None);
  
 # Compare the two dictionaries
   
 if SC1==CCP1 :
   return True
 else :
   return False;  
                   
def ReadClustersResults() :
  
  Dictionary={}; # Stores individual cluster data
  SC2D={}; # Stores data for the SerialCure 2D version
  
   # Traverse through all file paths to process cluster results
 
  Cluster_id=-1;
  for filepaths in FilePaths :
      for root,_,Files in os.walk(filepaths) :
        for files in Files :
          if "Results" in files : # Check for result files
           file_path = os.path.join(root,files);
           file_path1=re.search(r'/(.*?)Results',file_path);
           if file_path1 :
             file_path1=file_path1.group(1).strip();
             if "SingleCudaCure" in file_path : # Special handling for SingleCudaCure cases
                file_path1=re.search(r'/(.*)',file_path1);
                if file_path1 :
                    file_path1=file_path1.group(1).strip();
                    Dictionary["Type"]=file_path1;
             else:
                 Dictionary["Type"]=file_path1;
                            
           try :
               with open(file_path,"r") as file2 :  # Open and read the file
                 content=file2.readlines();
                 
                 # Extract various cluster properties
                 
                 for line in content :
                      NumberOfClusters=re.search(r'NumberOfClusters:(.*)',line);
                      if NumberOfClusters :
                         NumberOfClusters=NumberOfClusters.group(1).strip();
                         Dictionary["NumberOfClusters"]=int(NumberOfClusters);
                                                    
                      Cluster=re.search(r'Cluster:(.*?)item',line);   
                      if Cluster :
                         Cluster=Cluster.group(1).strip();
                         Dictionary[Cluster]={};
                         Cluster_id=Cluster;
                         
                         itemsize=re.search(r'item size:(.*?)Represantive',line);  
                         if itemsize :
                           itemsize=int(itemsize.group(1).strip());
                           Dictionary[Cluster]["item size"]=itemsize;
                         
                         RepresantivePoints=re.search(r'Represantive Points:(.*)',line);
                         if RepresantivePoints :
                            RepresantivePoints=int(RepresantivePoints.group(1).strip());
                            Dictionary[Cluster]["Number Of Represantive Points"]=RepresantivePoints;
                            
                      C=re.search(r'C:(.*)',line);
                      if C :
                         C=C.group(1).strip();
                         C=C.replace(" ",",");
                         CentrePoints=[float(points) for points in re.split(r',\s*',C)];
                         Dictionary[Cluster_id]["Centre Points"]=CentrePoints; 
                            
                      R=re.search(r'R:(.*)',line);
                      if R :
                         R=R.group(1).strip();
                         R=R.replace(" ",",");
                         RepresantivePoints=[float(points) for points in re.split(r',\s*',R)];
                         if Cluster_id in Dictionary and "Represantive Points" in Dictionary[Cluster_id] :
                            Dictionary[Cluster_id]["Represantive Points"].append(RepresantivePoints);   
                         else : 
                             Dictionary[Cluster_id]["Represantive Points"]=[];
                             Dictionary[Cluster_id]["Represantive Points"].append(RepresantivePoints);   
                             
                      CN=re.search(r'Cluster Nodes:(.*)',line);
                      if CN :
                         CN=CN.group(1).strip();
                         CN=CN.replace(" ",",");
                         ClusterNodes=[int(nodes) for nodes in re.split(r',\s*',CN)];
                         if Cluster_id in Dictionary and "Cluster Nodes" in Dictionary[Cluster_id] :
                            Dictionary[Cluster_id]["Cluster Nodes"].append(ClusterNodes);   
                         else : 
                            Dictionary[Cluster_id]["Cluster Nodes"]=ClusterNodes;     
                            
                # If it's a SerialCure result, store it in SC2D for comparison                  
               
               if "SerialCure" in Dictionary["Type"] :
                SC2D=Dictionary.copy();  
                
               # For other results, compare with SC2D 
               elif "AllDataGpu" in Dictionary["Type"] or "UnifiedMemory" in Dictionary["Type"] or "PrefetchAsyncUnifiedMemory" in Dictionary["Type"] or "PinnedMemoryCMH" in Dictionary["Type"] or "PinnedMemoryCHA" in Dictionary["Type"] or "MappedPinnedMemory" in Dictionary["Type"] :
                   if SameResults(SC2D,Dictionary) == True :
                     print("The results of",SC2D["Type"],"are same with the results of",Dictionary["Type"]);
                   else :
                     print("The results of",SC2D["Type"],"are not same with the results of",Dictionary["Type"]);
                     
                # For CudaCurePart results, plot and compare with SC2D     
                                           
               elif "CudaCurePart" in Dictionary["Type"] :
                if SameResults(SC2D,Dictionary) == True :
                  print("The results of",SC2D["Type"],"are same with the results of",Dictionary["Type"]);
                  global fig,ax1,ax2;
                  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5)); # Create side-by-side plots
                  printcluster(SC2D);  # Print SerialCure clusters
                  printcluster(Dictionary);  # Print CudaCurePart clusters
                  del SC2D;
                  Combine(); # Combine both plots
                else :
                  print("The results of",SC2D["Type"],"are not same with the results of",Dictionary["Type"]);  
          
               Dictionary.clear(); # Clear the dictionary after processing each file
                                                                                    
           except Exception as e :  # Handle file reading errors
            print("File Not Found",e);
            
                        
if __name__ == '__main__' :
     
  ReadClustersResults(); # Run the main function to process and compare result