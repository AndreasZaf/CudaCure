#define BUFSIZE 256
#define NONE (-2)

#ifndef MAXFLOAT
 #define   MAXFLOAT    ((double)3.40282346638528860e+38)
#endif

#ifdef READ_BINARY_FILE
/*#define SHIFT_FACTOR	100.0*/
#ifndef SHIFT_FACTOR
#define SHIFT_FACTOR	0.0
#endif
#else
#define SHIFT_FACTOR	0.0
#endif

#define FirstPruneRatio 0.3333
#define SecondPruneMulti 3

#define FirstCut 2
#define SecondCut 3

#define MAXBLOCKSIZE 512

static DATATYPE alpha = 0.3;		/* shrinking factor alpha */
static DATATYPE one_minus_alpha_rev;	/* unshrinking factor 1/(1-a) */

static int prune_clusters = 1;

static int blocksize=512,gridsize,new_gridsize;

static int size_limit = INT_MAX;

static int print_clusters=1;

static int nnpc=48024;
