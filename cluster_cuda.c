/*#define READ_BINARY_FILE*/

#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>
#include "alloc.h"
#include "error.h"
#include "cluster.h"
#include "Timers/Timers.h"
#include "help.h"
#include "CudaCure.h"

double my_gettime(void)
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return (double) (tv.tv_sec+tv.tv_usec/1000000.0);
}

#define BUFSIZE 256

typedef double my_time_t;

double distance(DATATYPE *pat1, DATATYPE *pat2, int lpat);
double root(double dist);
double cure_distance(struct BTree *x, struct BTree *y, int lpat);
void merge(struct BTree *item, int pair1, int pair2, int lpat, double min_dist);
void cluster(DATATYPE **pattern, DATATYPE **rep,int lpat,int npat,const char *type);
int read_pattern(FILE *Pfile, DATATYPE ***patternP, int lpattern, int npattern);
int read_centroids_pattern(FILE *Pfile, DATATYPE ***patternP, DATATYPE ***repP, int lpattern, int npattern);

static char buffer[BUFSIZE];		/* temporary string read buffer */

char   *program;			/* program name */

static int dflag = 0;			/* print distances */
static int width = 0;			/* width of ASCII tree representation */
static int norm = 2;			/* which l-norm to use for distance metric */

static int csize = 0; 			/* number of clusters */
static int rsize = 10;     /* number of rep points in a cluster */

static int threads = 1;
static int skip_patterns = 0;
static int real_lpat = 0;
static int load_centroids = 0;
static int input_nnbs = 0;
static int output_nnbs = 0;

static int read_binary_file = 0;

static int swap_bytes = 0;

static char nnb_inp_fname[80];
static char nnb_out_fname[80];

struct BTree *merge_tmp_item = NULL;
int *centroid_size;
int *centroid_rsize;
struct BTree *g_item = NULL;

FILE   *fp, *pfile, *outfp;
char  infname[80], outfname[80];
double *ydata;

void find_nnb(items, which, lpat, index, ndist)
  struct BTree *items;	/* node array */
  int which;		/* index of node to find nearest neightbor for */
  int lpat;		/* pattern length */
  int *index;		/* returns: index of nnb (-1 if none) */
  double *ndist;	/* returns: distance to nnb */
{
	int i;
	double dist, min_dist;
	int min_index;

	if (items[which].root == FALSE || items[which].full == TRUE) {
		*index = NONE;
		return;
	}

	min_index = NONE;
	min_dist = 0.0;
	/*
	 * find minimum distance neighbor -- to avoid duplication
	 * only pairs with 1st index < 2nd index are considered
	 */
	for (i = 0; i < which; i++) {
		if (items[i].root == FALSE || items[i].pruned == TRUE || items[i].full == TRUE)
			continue;

		dist = cure_distance(&items[which], &items[i], lpat);
		if (min_index == NONE || dist < min_dist) {
			min_index = i;
			min_dist = dist;
		}
	}

	*index = min_index;
	if (min_index >= 0)
		*ndist = min_dist;

	return;
}

static void usage()
{
	printf("usage: %s  [-n norm] [-k no-cluster] [-r rep-size] [-a alpha] [-p records] [-d dimension] [-c] [-u] [-s skip_patterns] [-f suffix] [-x] [-b] [-y] [-T SERIALCURE or PCUDACURE or ALLCUDACURE or UMCUDACURE or PAUMCUDACURE or CMHCUDACURE or CHACUDACURE or MCUDACURE   ] [vectorfile [namesfile]]\n", program);
  exit(2);
}

int main(int argc, char *argv[])
{
	FILE   *fp, *pfile;
  char *type=(char*)malloc(20*sizeof(char));
	char   *efile = NULL;
	char   *comps = NULL;
	DATATYPE **pattern = NULL;
	DATATYPE **rep = NULL;
  int lpat,npat;
	int	   has_suffix = 0;
	char   suffix[40];

	int   opt;
	extern char *optarg;
	extern int optind;

  int i;

	/* who are we ? */
	if ((program = strrchr(argv[0], '/')) == NULL)
		program = argv[0];
	else
		program += 1;

	while ((opt = getopt(argc, argv, "n:k:a:r:d:l:p:m:s:f:ucxgw:T:i:by")) != -1) {
		switch (opt) {
		case 'a':
	  	printf("alpha...\n");
	  	printf("optarg = %s\n", optarg);
	  	if (sscanf(optarg, DATATYPE_FMT, &alpha) != 1 || alpha < 0.0 || alpha > 1.0)
				usage();
	  	printf("alpha... = %f\n", alpha);
	  	break;
		case 'b':
		  read_binary_file = 1;
		  printf("read_binary_file = %d\n", read_binary_file);
		  break;
		case 'c':
		  load_centroids = 1;
		  printf("load_centroids = %d\n", load_centroids);
		  break;
		case 'd':
	  	if (sscanf(optarg, "%d", &lpat) != 1 || lpat < 0)
				usage();
	  	printf("lpat = %d\n", lpat);
	  	break;
		case 'f':
	  	if (sscanf(optarg, "%s", suffix) != 1)
				usage();
	  	has_suffix = 1;
	  	printf("suffix = %s\n", suffix);
			break;
		case 'g':
			dflag = 1;
			printf("dflag = %d\n", dflag);
			break;
		case 'i':
			strcpy(nnb_inp_fname, optarg);
			input_nnbs = 1;
			break;
		case 'k':
			if (sscanf(optarg, "%d", &csize) != 1 || csize < 0)
				usage();
			break;
		case 'l':
			if (sscanf(optarg, "%d", &real_lpat) != 1 || real_lpat < 0)
				usage();
			printf("real_lpat = %d\n", real_lpat);
			break;
		case 'm':
			if (sscanf(optarg, "%d", &size_limit) != 1 || size_limit < 0)
				usage();
			printf("size_limit = %d\n", size_limit);
			break;
		case 'n':
			if (sscanf(optarg, "%d", &norm) != 1 || norm < 0)
				usage();
			break;
		case 'p':
			if (sscanf(optarg, "%d", &npat) != 1 || npat < 0)
				usage();
			printf("npat = %d\n", npat);
			break;
		case 'r':
			if (sscanf(optarg, "%d", &rsize) != 1 || rsize < 0)
				usage();
			break;
		case 's':
			if (sscanf(optarg, "%d", &skip_patterns) != 1 || skip_patterns < 0)
				usage();
			printf("skip_patterns = %d\n", skip_patterns);
			break;

		case 'u':
	  	prune_clusters = 1;
	  	printf("prune_clusters = %d\n", prune_clusters);
	  	break;
		case 'w':
			strcpy(nnb_out_fname, optarg);
			output_nnbs = 1;
	  	break;
		case 'x':
	  	print_clusters = 1;
	  	printf("print_clusters = %d\n", print_clusters);
	  	break;
		case 'y':
	  	swap_bytes = 1;
	  	printf("swap_bytes = %d\n", swap_bytes);
	  	break;
    case 'T':
        if (sscanf(optarg,"%s",type) != 1 || (strcmp(type,"SERIALCURE") != 0  && strcmp(type,"PCUDACURE") != 0 && strcmp(type,"ALLCUDACURE") != 0 && strcmp(type,"UMCUDACURE") != 0 && strcmp(type,"PAUMCUDACURE") != 0 && strcmp(type,"CMHCUDACURE") != 0 && strcmp(type,"CHACUDACURE") != 0 && strcmp(type,"MCUDACURE") != 0) || strlen(type) == 0)
           usage(); 
           break;        
		case '?':
	  	usage();
	  	break;
		}
	}

	if (alpha != 1.0)
		one_minus_alpha_rev = 1.0 / (1.0 - alpha);
	else {
		one_minus_alpha_rev = 0.0;	/* it's a solution. However it makes no sense, that's why..*/
		if (rsize != 1) {
			printf("WARNING: r = %d and a = %f. Setting r = 1\n", rsize, alpha);
			rsize = 1;
		}
	}
 

	printf("alpha = %f  rsize = %d\n", alpha, rsize);

	if (real_lpat == 0)	real_lpat = lpat;

	if (optind + 2 < argc)
		usage();

	if (!(optind < argc) || !strcmp(argv[optind], "-"))
		fp = stdin;
	else
		IfErr(fp = fopen(argv[optind], "r")) {
			printf( "%s: cannot open file %s\n", argv[0], argv[optind]);
			exit(1);
	}

	strcpy(infname, argv[optind]);
	strcpy(outfname, infname);
	if (has_suffix)
	{
		strcat(outfname, ".");
		strcat(outfname, suffix);
	}
	else
	{
		char tmpbuffer[80];
		sprintf(tmpbuffer, ".%d", getpid());
		strcat(outfname, tmpbuffer);
	}
	strcat(outfname, ".out");

	if (!load_centroids) {
		IfErr(read_pattern(fp, &pattern, lpat, npat)) {
			printf( "%s: cannot read pattern\n", program);
			exit(1);
		}
	}
	else {
		IfErr(read_centroids_pattern(fp, &pattern, &rep, lpat, npat)) {
			printf( "%s: cannot read pattern\n", program);
			exit(1);
		}
	}

	printf("pattern = %p, npat = %d, lpat = %d\n", pattern, npat, lpat);
	printf( "read %d patterns:  size = %d\n", npat, lpat);

  if(strcmp(type,"SERIALCURE")==0)
 {     
   cluster(pattern, rep,lpat,npat,"SERIALCURE");
   printf("Serial Cure clustering done\n");
 }   
 
  if(strcmp(type,"PCUDACURE") == 0 || strcmp(type,"ALLCUDACURE") == 0 || strcmp(type,"UMCUDACURE") == 0 || strcmp(type,"PAUMCUDACURE") == 0 || strcmp(type,"CMHCUDACURE") == 0 || strcmp(type,"CHACUDACURE") == 0  || strcmp(type,"MCUDACURE") == 0 )
 {  
	 cluster(pattern, rep,lpat,npat,type);
	 printf("Cuda Cure clustering done\n");
 } 
	return 0;
}


void cluster(DATATYPE **pattern, DATATYPE **rep,int lpat,int npat,const char *type)
{
	int i, j, k, cno;
  
  int nodes=npat;
  
  double gmin_dist;
  int gpair1;
  
  double init_timer,find_mdp_timer,clustering_timer,merge_timer,update_timer,pruning_timer,kernel_timer;
  
  int chunk;
   
	my_time_t t1, t2, t11, t22;
	my_time_t curr_t = 0.0;
 
  my_time_t init_leaf_nodes_time = 0.0; 
  
  my_time_t start_cluster_t,stop_cluster_t;

  struct BTree *item;
  struct nnb_info *nnb;
  
  int pruned_nodes=0;
  int times_here=0;
  
      
	int irep;

	int u_nnb;
	int u_nnb_idx[1024];

	/*
	 * for each data point or cluster center, we keep the index of the nearest
	 * neighbor, as well as the distance to it.
	 */

	if (posix_memalign((void **)&nnb, 4096, npat*sizeof(struct nnb_info))) {
		printf( "%s: not enough core\n", program);
		exit(1);
	}

	if (posix_memalign((void **)&item, 4096, npat*sizeof(struct BTree))) {
		printf( "%s: not enough core\n", program);
		exit(1);
	}

	g_item = item;	/* used by evaluate() */

	/*
	 * initialize leaf nodes
	 */
	merge_tmp_item = calloc(1, sizeof(struct BTree));
	IfErr (merge_tmp_item->mean = calloc(lpat, sizeof(DATATYPE))) {
		printf( "%s: not enough core\n", program);
		exit(1);
	}
	IfErr (merge_tmp_item->rep = calloc(2*rsize, sizeof(DATATYPE *))) {
		printf( "%s: not enough core\n", program);
		exit(1);
	}
	for (j=0; j < 2*rsize; ++j) {
		IfErr (merge_tmp_item->rep[j] = calloc(lpat, sizeof(DATATYPE))) {
			printf( "%s: not enough core\n", program);
			exit(1);
		}
	}

	t1 = my_gettime();
	irep = 0;
	for (i = 0; i < npat; i++) {
#if 0	/* memory already allocated */
		item[i].mean = pattern[i];
#else	/* keep unmodified the patterns (input data) */
		item[i].mean = malloc(lpat*sizeof(DATATYPE));
		memcpy(item[i].mean, pattern[i], lpat*sizeof(DATATYPE));
#endif

		item[i].root = TRUE;
		item[i].size = 1;
		item[i].full = FALSE;
		item[i].pats = calloc(1, nnpc*sizeof(int));	/* xxx */
		item[i].pats[0] = i;
		if (load_centroids)
			item[i].size = centroid_size[i];	/* ... */
		item[i].distance = 0.0;

		IfErr (item[i].rep = calloc(rsize, sizeof(DATATYPE *))) {
		  printf( "%s: not enough core\n", program);
		  exit(1);
		}
#if 1		/* Optimization.. */
		if ((rsize == 1)&&(alpha == 1.0)) {
			item[i].rep[0] = item[i].mean;
		}
		else
#endif
		{
			if (!load_centroids) {
				for (j=0; j < rsize; ++j) {	/* optimized: only one allocation is required (in fact even less)...*/
					if (j == 0) {
						IfErr (item[i].rep[j] = calloc(lpat, sizeof(DATATYPE))) {
							printf( "%s: not enough core\n", program);
							exit(1);
						}
					}
					else {
						item[i].rep[j] = NULL;
					}
				}
				memcpy(item[i].rep[0], item[i].mean, lpat*sizeof(DATATYPE));
			}
			else {
				for (k = 0; k < rsize; k++) {
					/*memcpy(item[i].rep[k], rep[irep], lpat*sizeof(DATATYPE));*/
					/*free(rep[irep]);*/
					item[i].rep[k] = rep[irep];	/* already allocated */
				        irep++;
				}
			}
		}

	}

	t2 = my_gettime();
	init_leaf_nodes_time = t2 - t1;
 
 
	/*
	 * initialize nearest neighbors
	 */
    
    if(strcmp(type,"ALLCUDACURE")==0 || strcmp(type,"UMCUDACURE")==0 || strcmp(type,"PAUMCUDACURE")==0 || strcmp(type,"CMHCUDACURE")==0 || strcmp(type,"CHACUDACURE")==0 || strcmp(type,"MCUDACURE")==0 || strcmp(type,"MALLCUDACURE")==0)   
    {   
      Gpu_Single_Clustering(item,npat,rsize,norm,lpat,csize,type);
  
   }
   
   if(strcmp(type,"SERIALCURE")==0)
   {
       
       printf("--------------------------------------------------------------SERIALCURE-------------------------------------------------------------------------\n\n");
     
   }else if(strcmp(type,"PCUDACURE")==0)
    {
     
     if(npat>1500)
     { 
       printf("Large Data for CudaCurePart\n");
       exit(1);
      }
     printf("---------------------------------------------------------------CUDACUREPART-------------------------------------------------------------------------\n\n");
     
    }

 if(strcmp(type,"SERIALCURE")==0)
 {

	t1 = my_gettime();

	if (input_nnbs) {
		FILE *nnb_fd;

		nnb_fd = fopen(nnb_inp_fname, "r");
		for (i = 0; i < npat; i++) {
			int res;
			res = fscanf(nnb_fd, "%d\t%lf\n", &nnb[i].index, &nnb[i].dist);
		}
		fclose(nnb_fd);
   
   	t2 = my_gettime();
	  init_timer = t2 - t1;
	  printf("GPU INIT NNB TIME = %f\n",init_timer); fflush(0);
   
	}
	else {
 
     for(unsigned int i=0; i<npat; i++)
     {
        find_nnb(item, i, lpat, &nnb[i].index, &nnb[i].dist);
     }
     
	}
  
  t2 = my_gettime();
  
  init_timer=t2-t1; 
  
 }else if(strcmp(type,"PCUDACURE")==0)
 {
  kernel_timer=0.0;
  GPU_initialization_phase(item,nnb,npat,rsize,norm,lpat,&kernel_timer);
  init_timer=kernel_timer;
 }
  
	if (output_nnbs) {
		FILE *nnb_fd;

		nnb_fd = fopen(nnb_out_fname, "w");
		for (i = 0; i < npat; i++) {
			fprintf(nnb_fd, "%d\t%d\t%.10f\n", i, nnb[i].index, nnb[i].dist);
		}
		fclose(nnb_fd);
		exit(0);
	}


	/* XXX: The first min distance pair can be computed in the init phase.
		However, using the following code with the scheduling policy of the
		update phase allows for a first-touch of the pages that correspond to
		the nnb (dist,index) structure */

	/*
	 * find minimum distance pair
	 */
   
  if(strcmp(type,"SERIALCURE")==0)
  { 
  
 	  t11 = my_gettime();
    
   	gpair1 = NONE;
	  gmin_dist = -0.1;
  
		for (i = 0; i < npat; i++) {
			if (item[i].root == FALSE || item[i].pruned == TRUE || item[i].full == TRUE) {
				continue;
			}
			if (nnb[i].index != NONE && (gpair1 == NONE || nnb[i].dist < gmin_dist)) {
				gpair1 = i;
				gmin_dist = nnb[i].dist;

			}
		}	
   		
	t22 = my_gettime();	
	find_mdp_timer += (t22-t11);
  
  }else if(strcmp(type,"PCUDACURE")==0)
  {
   kernel_timer=0.0;
   GPU_minimum_distance_pair(item,nnb,npat,&gpair1,&gmin_dist,&kernel_timer);
   find_mdp_timer=kernel_timer;

  }       
	/*
	 * cluster until done
	 */

if(strcmp(type,"SERIALCURE")==0)
{

  printf("ENTERING CLUSTERING PHASE (nodes=%d, clusters=%d)\n", nodes, csize); fflush(0);
  
	start_cluster_t = my_gettime();	
	while (nodes > csize) 
  {
  
		struct BTree  *newitem;
   
    double dist,min_dist;   
    int pair1, pair2;
   
		 times_here++;
   
 		 pair1 = gpair1;
	   pair2 = nnb[pair1].index;
	   min_dist = gmin_dist;
       
		 if (pair1 == NONE)
			break;		/* analysis finished */

		 min_dist = root(min_dist);
      
		 if ((dflag) && (times_here % 1000 == 0)) {
			if (curr_t == 0.0) curr_t = my_gettime();
			printf("[%3d] [%5d] minimum distance = %.10f (%f) - (%f)\n", times_here, nodes, (double)min_dist, my_gettime()-start_cluster_t, my_gettime()-curr_t); fflush(NULL);
			curr_t = my_gettime();
		 }
   
		 t11 = my_gettime();
		 merge(item, pair1, pair2, lpat, min_dist);
		 t22 = my_gettime();
		 merge_timer += (t22-t11);
      
 		 t11 = my_gettime();
		 chunk = 10;

		 gmin_dist = -0.1;
		 gpair1 = NONE;
     
   		for (i = 0; i < npat; i++) {
			if (nnb[i].index == NONE || item[i].root == FALSE || 	item[i].pruned == TRUE || item[i].full == TRUE) {
				continue;
			} else if (i < pair2) {
				if ((gpair1 == NONE) || (nnb[i].dist < gmin_dist)) {
					gpair1 = i;
					gmin_dist = nnb[i].dist;
                     
				}
			} else if (nnb[i].index == pair1 || nnb[i].index == pair2) {
        
				find_nnb(item, i, lpat, &nnb[i].index, &nnb[i].dist);
				/* XXX: A new index is computed, which can be NONE if there is no valid entry with smaller index */
				if ((nnb[i].index != NONE) && ((gpair1 == NONE) || (nnb[i].dist < gmin_dist))) {
					gpair1 = i;
					gmin_dist = nnb[i].dist;
                     
				}
			} else if (i < pair1) {
      
      
				if ((gpair1 == NONE) || (nnb[i].dist < gmin_dist)) {
					gpair1 = i;
					gmin_dist = nnb[i].dist;
                    
                    
				}
			} else if (pair1 < i) {
         
				dist = cure_distance(&item[pair1], &item[i], lpat);
				if (dist < nnb[i].dist) {
					/*
				 	* distance to new node is smaller than previous nnb,
				 	* so make it the new nnb.
				 	*/
					nnb[i].index = pair1;
					nnb[i].dist = dist;
				}
				if ((gpair1 == NONE) || (nnb[i].dist < gmin_dist)) {
					gpair1 = i;
					gmin_dist = nnb[i].dist;
                              
				}
			}
		}	/* for */
  

		t22 = my_gettime();
		update_timer += (t22-t11); 
   
   --nodes;
   
   	if (prune_clusters) {
 			  if (nodes == (int)(npat * FirstPruneRatio)) {
				printf("==== First phase of pruning at %d nodes remaining ====\n", nodes);
        
				t11 = my_gettime();

				for (i = 0; i < npat; i++) {
					if (item[i].root == TRUE && item[i].size < FirstCut) {
/*						printf("[%d] ====    %d of size %d removed\n", omp_get_thread_num(), i, item[i].size);*/
						pruned_nodes++;
						item[i].pruned = TRUE;
						--nodes;
					}
				}

				t22 = my_gettime(); pruning_timer += (t22-t11);
				printf("first pruning - (1): %lf seconds (pruned nodes = %d)\n", t22-t11, pruned_nodes);
        
				 t11 = my_gettime();
        
				 gpair1 = NONE;
				 gmin_dist = -0.1;

					for (i = 0; i < npat; i++) {
						if (item[i].root == TRUE && item[i].full == FALSE && item[nnb[i].index].pruned == TRUE)
      
							find_nnb(item, i, lpat, &nnb[i].index, &nnb[i].dist);
                                      
						if (item[i].root == FALSE || item[i].pruned == TRUE || item[i].full == TRUE) {
              
							continue;
						}
						if (nnb[i].index != NONE && (gpair1 == NONE || nnb[i].dist < gmin_dist)) {
							gpair1 = i;
							gmin_dist = nnb[i].dist;
						}
					} /* for */

					
				t22 = my_gettime(); pruning_timer += (t22-t11);
				printf("first pruning - (2): %lf seconds\n", t22-t11);
        
        
        }else if (nodes == csize * SecondPruneMulti) { 
        
				   printf("==== Second phase of pruning at %d nodes remaining ====\n", nodes);
                        
				   t11 = my_gettime();

				  for (i = 0; i < npat; i++) {
					  if (item[i].root == TRUE && item[i].full == FALSE && item[i].pruned == FALSE && item[i].size < SecondCut) {
						 item[i].pruned = TRUE;
						 pruned_nodes++;
						 --nodes;
					 }
				 }
           
				 t22 = my_gettime();	pruning_timer += (t22-t11);
				 printf("second pruning - (1): %lf seconds - pruned nodes = %d\n", t22-t11, pruned_nodes);
                         
          t11 = my_gettime();
        
				 gpair1 = NONE;
				 gmin_dist = -0.1;


					for (i = 0; i < npat; i++) {
						if (item[i].root == TRUE && item[i].full == FALSE && item[nnb[i].index].pruned == TRUE)
							find_nnb(item, i, lpat, &nnb[i].index, &nnb[i].dist);

						if (item[i].root == FALSE || item[i].pruned == TRUE || item[i].full == TRUE) {
							continue;
						}
						if (nnb[i].index != NONE && (gpair1 == NONE || nnb[i].dist < gmin_dist)) {
							gpair1 = i;
							gmin_dist = nnb[i].dist;
						}
					} /* for*/
					
				t22 = my_gettime(); pruning_timer += (t22-t11);
				printf("second pruning - (2): %lf seconds\n", t22-t11); 
        
        
        }    
    }
  }
  
 	stop_cluster_t = my_gettime();
	clustering_timer = stop_cluster_t - start_cluster_t;
 
  Gpu_Cuda_Cure_Part_Results(npat,lpat,csize,init_timer,find_mdp_timer,clustering_timer,merge_timer,update_timer,pruning_timer,"SerialCure");
      
  Gpu_Cuda_Cure_Print_Results(item,npat,rsize,lpat,nodes,"SerialCure");
   

}else if(strcmp(type,"PCUDACURE")==0)
{
	printf("ENTERING CLUSTERING PHASE (nodes=%d, clusters=%d)\n", nodes, csize); fflush(0);
      
	start_cluster_t = my_gettime();	
	while (nodes > csize) {
	struct BTree  *newitem;
   
   int pair1,pair2;
   
   double dist,min_dist;   
   
		times_here++;
   
 		pair1 = gpair1;
    
	  pair2 = nnb[pair1].index;
       
	  min_dist = gmin_dist;
     
		if (pair1 == NONE)
			 break;		/* analysis finished */
    
		min_dist = root(min_dist);
      
		if ((dflag) && (times_here % 1000 == 0)) {
			if (curr_t == 0.0) curr_t = my_gettime();
			printf("[%3d] [%5d] minimum distance = %.10f (%f) - (%f)\n", times_here, nodes, (double)min_dist, my_gettime()-start_cluster_t, my_gettime()-curr_t); fflush(NULL);
			curr_t = my_gettime();
		}
   
    
		t11 = my_gettime();
		merge(item, pair1, pair2, lpat, min_dist);
		t22 = my_gettime();
		merge_timer += (t22-t11);
    
    chunk = 10;
 
  kernel_timer=0.0;
  GPU_update_nnb(item,nnb,npat,rsize,norm,lpat,pair1,pair2,&gpair1,&gmin_dist,&kernel_timer);
  update_timer+=kernel_timer;


//		printf("-\n");

   
		/* number of nodes have been reduced */
   
		--nodes;
   
		/*
		 * prune clusters based on their size - must be optimized (loops to be merged)
		 */
		if (prune_clusters) {
			if (nodes == (int)(npat * FirstPruneRatio)) {
				printf("==== First phase of pruning at %d nodes remaining ====\n", nodes);
        
            
      kernel_timer=0.0;
      GPU_first_pruning(item,npat,&nodes,&pruned_nodes,&kernel_timer); 
      pruning_timer +=kernel_timer;
			printf("GPU first pruning - (1): %lf seconds (pruned nodes = %d)\n",kernel_timer, pruned_nodes);
      
      
      kernel_timer=0.0;
      GPU_pruning(item,nnb,npat,rsize,norm,lpat,&gpair1,&gmin_dist,&kernel_timer);
      pruning_timer +=kernel_timer;
      printf("GPU first pruning - (2): %lf seconds\n",kernel_timer);
        
			}
			else if (nodes == csize * SecondPruneMulti) {
				printf("==== Second phase of pruning at %d nodes remaining ====\n", nodes);
        
        
           kernel_timer=0.0;
           GPU_second_pruning(item,npat,&nodes,&pruned_nodes,&kernel_timer);
           pruning_timer +=kernel_timer;
				   printf("GPU second pruning - (1): %lf seconds - pruned nodes = %d\n",kernel_timer, pruned_nodes);
                
           kernel_timer=0.0;
           GPU_pruning(item,nnb,npat,rsize,norm,lpat,&gpair1,&gmin_dist,&kernel_timer);
           pruning_timer +=kernel_timer;
           printf("GPU second pruning - (2): %lf seconds\n",kernel_timer);

			}
		}

	}	/* while */

	stop_cluster_t = my_gettime();
	clustering_timer = stop_cluster_t - start_cluster_t;
    
   Gpu_Cuda_Cure_Part_Results(npat,lpat,csize,init_timer,find_mdp_timer,clustering_timer,merge_timer,update_timer,pruning_timer,"CudaCurePart");
      
   Gpu_Cuda_Cure_Print_Results(item,npat,rsize,lpat,nodes,"CudaCurePart");

}

	return;
}

 
double cure_distance(struct BTree *x, struct BTree *y, int lpat)
{
        register int i, j;
        double min_dist = MAXFLOAT, dist;
        int xmax, ymax;

        if ((x->full == TRUE) || (y->full == TRUE)) {
                return 1e99; //DBL_MAX;
                //printf("why (%d %d)?\n", x->size, y->size);
                //exit(1);
        }

        if (x->size > rsize)
                xmax = rsize;
        else
                xmax = x->size;

        if (y->size > rsize)
                ymax = rsize;
        else
                ymax = y->size;

        for (i=0; i<xmax; ++i)
                for (j=0; j<ymax; ++j)
                        if ((dist = distance(x->rep[i], y->rep[j], lpat)) < min_dist)
                                min_dist = dist;

  return min_dist;
}

double cure_pat_cluster_distance(DATATYPE *p, struct BTree *y, int lpat)
{
	register int i, j;
	double min_dist = MAXFLOAT, dist;
	int xmax, ymax;

	if (y->size > rsize)
		ymax = rsize;
	else
		ymax = y->size;

	if ((dist = distance(p, y->mean, lpat)) < min_dist)
			min_dist = dist;

	for (j=0; j<ymax; ++j)
		if ((dist = distance(p, y->rep[j], lpat)) < min_dist)
			min_dist = dist;

	return min_dist;
}

double distance(DATATYPE *pat1, DATATYPE *pat2, int lpat)
{
	register int i;
	double   dist = 0.0;

	for (i = 0; i < lpat; i++) {
		double   diff = 0.0;
                 
#if 0
#ifndef NO_DONTCARES
		if (!IS_DC(pat1[i]) && !IS_DC(pat2[i]))
#endif
#endif
		diff = pat1[i] - pat2[i];

		switch (norm) {
		double   adiff;

		case 2:
			dist += diff * diff;
			break;
		case 1:
			dist += fabs(diff);
			break;
		case 0:
			if ((adiff = fabs(diff)) > dist)
			dist = adiff;
			break;
		default:
			dist += pow(fabs(diff), (double) norm);
			break;
		}
	}

	return dist;
}

double root(double dist)
{
	if (dist == 0) return 0;

	switch (norm) {
	case 2:
		return sqrt(dist);
	case 1:
	case 0:
		return dist;
	default:
		return pow(dist, 1 / (double) norm);
	}
}

#undef isnan
#define isnan(x)	(0)

double read_nextnum(FILE *fp)
{
	double val;

	fscanf(fp, "%lf", &val);
	return val;
}

int read_pattern(FILE *Pfile, DATATYPE ***patternP, int lpattern, int npattern)
{
	register int i;
	int   status;
	int	real_lpattern = real_lpat;

	/***** these local variables are temporary storage and are ****
	 ***** copied to the real variables if there is no error ******/
	int   ipattern;
	DATATYPE **pattern;

  /* allocate space for input/target arrays */
	IfErr(pattern = new_2d_array_of(npattern, lpattern, DATATYPE)) {
		printf("cannot allocate memory for patterns\n");
		exit(1);
	}

#if 1
	ydata = malloc(npattern*sizeof(DATATYPE));
	printf("npattern = %d ydata = %p\n", npattern, ydata);
#if 0
	FILE *fpin;
	fpin = open_traindata();
	printf("fpin = %p\n", fpin);
#else
	FILE *fpin = Pfile;
#endif

	for (ipattern = 0; ipattern < npattern; ipattern++) {
		for (i = 0; i < real_lpattern; i++) {
			pattern[ipattern][i] = read_nextnum(fpin);
		}
#if defined(SURROGATES)
#if 0
		ydata[ipattern] = read_nextnum(fpin);
#else
		ydata[ipattern] = 0.0;
#endif
#else
		ydata[ipattern] = 0.0;
#endif
#if 0
		printf("pat %d: ", ipattern);
		for (i = 0; i < real_lpattern; i++) {
			printf("%f ", pattern[ipattern][i]);
		}
		printf(" : %f", ydata[i]); 
		printf("\n");
#endif
	}
	fclose(fpin);
	goto label100;
#endif


label100:
	/* free any space already allocated for patterns */
	if (*patternP != NULL)
		free_2d_array((char **)*patternP, npattern);

	*patternP = pattern;	/* input array */

	return MY_OK;		/* patterns were read in without error */
}

int read_centroids_pattern(FILE *Pfile, DATATYPE ***patternP, DATATYPE ***repP, int lpattern, int npattern)
{
  register int i;
  int   status;

  /***** these local variables are temporary storage and are ****
   ***** copied to the real variables if there is no error ******/
  int   ipattern;
  DATATYPE **pattern;
  DATATYPE **rep;
  int  irep, k;

  /* allocate space for input/target arrays */
  IfErr(pattern = new_2d_array_of(npattern, lpattern, DATATYPE)) {	/* this is for the centroid */
	fprintf(stderr, "cannot allocate memory for patterns\n");
	exit(1);
  }

  if ((rsize == 1)&&(alpha == 1.0)) {
  	/* do nothing: Centroid == Representative point */
  }
  else
    IfErr(rep = new_2d_array_of(npattern*rsize, lpattern, DATATYPE)) {	/* this is for the centroid */
		fprintf(stderr, "cannot allocate memory for patterns\n");
		exit(1);
	}

  /**** this loop reads in one line from pattern file,**
   **** stores each pattern into pattern buffer  *****/

  centroid_size = malloc(npattern*sizeof(int));
  if (centroid_size == NULL) {
	printf("cannot allocate memory for centroid_size\n");
	exit(1);
  }

  centroid_rsize = malloc(npattern*sizeof(int));
  if (centroid_rsize == NULL) {
	printf("cannot allocate memory for centroid_rsize\n");
	exit(1);
  }

  irep = 0;
  for (ipattern = 0; ipattern < npattern; ipattern++) {
#if 1 /*ndef READ_BINARY_FILE */
	float f;
	int integer;

	/* read centroid no */
	IfErr (status = fscanf(Pfile, "%s", buffer)) {
		fprintf(stderr,"cannot read centroid # %d\n", ipattern);
		exit(1);
	}
	IfErr (status = sscanf(buffer, "%d", &integer)) {
		fprintf(stderr,"cannot read centroid # %d", ipattern);
		exit(1);
	}

	/* read size */
	IfErr (status = fscanf(Pfile, "%s", buffer)) {
		fprintf(stderr,"cannot read size for centroid # %d", ipattern);
		exit(1);
	}
	IfErr (status = sscanf(buffer, "%d", &integer)) {
		fprintf(stderr,"cannot read size for centroid # %d", ipattern);
		exit(1);
	}

	centroid_size[ipattern] = integer;

	/* read rsize */
	IfErr (status = fscanf(Pfile, "%s", buffer)) {
		fprintf(stderr,"cannot read size for centroid # %d", ipattern);
		exit(1);
	}
	IfErr (status = sscanf(buffer, "%d", &integer)) {
		fprintf(stderr,"cannot read size for centroid # %d", ipattern);
		exit(1);
	}

	centroid_rsize[ipattern] = integer;


	/* read mean */
	/* C: */
	IfErr (status = fscanf(Pfile, "%s", buffer)) {
		fprintf(stderr,"cannot read pattern # %d", ipattern);
		exit(1);
	}

	for (i = 0; i < lpattern; i++) {
		IfErr (status = fscanf(Pfile, "%s", buffer)) {
			fprintf(stderr,"cannot read pattern # %d", ipattern);
			exit(1);
		}
		IfErr (status = sscanf(buffer, "%f", &f)) {
			fprintf(stderr,"cannot read pattern # %d", ipattern);
			exit(1);
		}
		pattern[ipattern][i] = f + SHIFT_FACTOR;
	}

	/* read representative points */
	if ((rsize == 1)&&(alpha == 1.0)) {
  		/* Centroid == Representative point : Just read the rep point and ignore it */
		IfErr (status = fscanf(Pfile, "%s", buffer)) {
			fprintf(stderr,"cannot read pattern # %d", ipattern);
			exit(1);
		}

		for (i = 0; i < lpattern; i++) {
			IfErr (status = fscanf(Pfile, "%s", buffer)) {
				fprintf(stderr,"cannot read pattern # %d", ipattern);
				exit(1);
			}
			IfErr (status = sscanf(buffer, "%f", &f)) {
				fprintf(stderr,"cannot read pattern # %d", ipattern);
				exit(1);
			}
			/*rep[irep][i] = f + SHIFT_FACTOR;*/
		}
  	}
	else {
		for (k = 0; k < rsize; k++) {
			if (k < centroid_size[ipattern]) {
				IfErr (status = fscanf(Pfile, "%s", buffer)) {
					fprintf(stderr,"cannot read pattern # %d", ipattern);
					exit(1);
				}
		
				for (i = 0; i < lpattern; i++) {
					IfErr (status = fscanf(Pfile, "%s", buffer))  {
						fprintf(stderr,"cannot read pattern # %d", ipattern);
						exit(1);
					}
					IfErr (status = sscanf(buffer, "%f", &f)) {
						fprintf(stderr,"cannot read pattern # %d", ipattern);
						exit(1);
					}
					rep[irep][i] = f + SHIFT_FACTOR;
				}
			}
			else {
				/* not used */
			}
			irep++;		
		}
	}
#else
	float f;
	IfErr (status =  fread(&f, 4, 1, Pfile)) {
		fprintf(stderr,"cannot read pattern # %d", ipattern);
		exit(1);
	}
	for (i = 0; i < lpattern; i++) {
		IfErr (status = fread(&f, 4, 1, Pfile)) {
			fprintf(stderr,"cannot read pattern # %d", ipattern);
			exit(1);
		}
		pattern[ipattern][i] = f + SHIFT_FACTOR;
	}
#endif
#if 0
	printf("pat %d", ipattern);
	for (i = 0; i < lpattern; i++) {
		printf("%f ", pattern[ipattern][i]);
	}
	printf("\n");
#endif
	}
	/* if there is any error, these pointers below aren't changed */

	/* free any space already allocated for patterns */
	if (*patternP != NULL)
		free_2d_array((char **)*patternP, npattern);

	*patternP = pattern;	/* input array */

	if (*repP != NULL)
		free_2d_array((char **)*repP, npattern);

	*repP = rep;	/* input array */

	return MY_OK;		/* patterns were read in without error */
}

void merge(struct BTree *item, int pair1, int pair2, int lpat, double min_dist)
{
	register int i, j, k;
	struct BTree *newitem;
	int newsize = item[pair1].size+item[pair2].size;
	DATATYPE r1 = (DATATYPE)item[pair1].size/newsize;
	DATATYPE r2 = (DATATYPE)item[pair2].size/newsize;
	double maxDist;
	DATATYPE *repPoint;
	double minDist, dist;
	int max_i;
	int rcount;
	int pair1_npoints, pair2_npoints, merged_npoints;
	static	int point_used[64];


	pair1_npoints = (rsize <= item[pair1].size)? rsize: item[pair1].size;
	pair2_npoints = (rsize <= item[pair2].size)? rsize: item[pair2].size;

/*	printf("merging %d (%d/%d) and %d (%d/%d)\n", pair1, item[pair1].size, pair1_npoints, pair2, item[pair2].size, pair2_npoints);*/

	/* unshrink representative points */
	if ((item[pair1].size > rsize)&&(alpha < 1.0)) {
		for (j=0; j<rsize; ++j)
			for (k=0; k<lpat; ++k)
				item[pair1].rep[j][k] = one_minus_alpha_rev*(item[pair1].rep[j][k] - alpha*item[pair1].mean[k]);
        
	}
 
	if ((item[pair2].size > rsize)&&(alpha < 1.0)) {
		for (j=0; j<rsize; ++j)
			for (k=0; k<lpat; ++k)
				item[pair2].rep[j][k] = one_minus_alpha_rev*(item[pair2].rep[j][k] - alpha*item[pair2].mean[k]);
       
	}
 

	/*
	 * replace left child node with new tree node
	 * link right child node into parent
	 */

	/* set the size and clear up some memory */
	merge_tmp_item->size = item[pair1].size;
	merge_tmp_item->distance = item[pair1].distance;

	int l, m;
	k = item[pair1].size;
	l = item[pair2].size;
	for (m = 0; m < l; m++)
		item[pair1].pats[m+k] = item[pair2].pats[m];

	item[pair1].size = newsize;
	if (newsize > size_limit) item[pair1].full = TRUE;
	item[pair1].distance = min_dist;

	item[pair2].root = FALSE;	/* jth item is no longer a root. it's a subtree */

	/* find mean of two clusters as weighted average*/
	for (i=0; i<lpat; ++i)
		item[pair1].mean[i] = r1*item[pair1].mean[i] + r2*item[pair2].mean[i];
   
   

/*
	printf("merge mean : %f %f\n", item[pair1].mean[0], item[pair1].mean[1]);
*/
	/* find new representative points */

	if (newsize <= rsize) {
		for (i=pair1_npoints, j = 0; j < pair2_npoints; ++i, ++j) {
			item[pair1].rep[i]  = item[pair2].rep[j];
		}
	}
	else {	/* newsize > rsize: selection is required */
		/* store representatives of pair1 somewhere else */

		/* store all represenative in a temporary array */
		for (i=0; i < pair1_npoints; ++i) {
			memcpy(merge_tmp_item->rep[i], item[pair1].rep[i], lpat*sizeof(DATATYPE));
		}
		for (i=pair1_npoints, j = 0; j < pair2_npoints; ++i, ++j) {
			memcpy(merge_tmp_item->rep[i], item[pair2].rep[j], lpat*sizeof(DATATYPE));	/* for pair2, I can have an assignment for i >= rsize !!! */
													/* i.e, merge_tmp_item->rep[i] = item[pair2].rep[j]; */
		}
   
   

		/* "allocate" space for new representative points of pair1 - we do not free anything...*/
		for (i=pair1_npoints, j = 0; (i < rsize) && (j < pair2_npoints); ++i, ++j) {
			/* if (i < rsize) */
			item[pair1].rep[i] = item[pair2].rep[j];
			/* else
			free(item[pair2].rep[j]);
			*/
		}
   
    
		for (i = 0; i < (pair1_npoints+pair2_npoints); i++) point_used[i] = 0;

		for (rcount=0; rcount < rsize; ++rcount) {
			int point_id = -1;
			maxDist = -0.1;
			for (i = 0; i < (pair1_npoints+pair2_npoints); i++) {
				if (point_used[i]) continue;
				if (rcount == 0) {
					dist = distance(merge_tmp_item->rep[i], item[pair1].mean, lpat);
               
            
					if (dist > maxDist) {
						maxDist = dist;
						repPoint = merge_tmp_item->rep[i];
						point_id = i;
					}
				}
				else {
					dist = distance(merge_tmp_item->rep[i], item[pair1].rep[rcount-1], lpat);
               
           
					if (dist > maxDist) {
						maxDist = dist;
						repPoint = merge_tmp_item->rep[i];
						point_id = i;
					}
				}
			}
			point_used[point_id] = 1;
			/* printf("pair1 = %d rcount = %d repPoint = %p\n", pair1, rcount, repPoint);*/
			memcpy(item[pair1].rep[rcount], repPoint, lpat*sizeof(DATATYPE));
		}
   
 

		/* shrink based on alpha */
		/*	if (alpha == 1.0);*/
		for (j=0; j< rsize; ++j)
			for (k=0; k<lpat; ++k)
				item[pair1].rep[j][k] += (alpha*(item[pair1].mean[k]-
					         item[pair1].rep[j][k]));
                                                                                
                                                                                
                                                                                
                                                                                                                                                                                                                                                                                                                                                                        
	}
 
}