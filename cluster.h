/*
 * cluster.h -- declarations for clustering routines
 *
 */

#include <stdio.h>

#if 1
#define DATATYPE	double
#define DATATYPE_FMT	"%lf"
#else
#define DATATYPE	float
#define DATATYPE_FMT	"%f"
#endif

 /* 32 bytes -> 2 records for alignment on 64 bytes boundaries */
struct BTree {
#ifdef ORIGINAL_VERSION
	DATATYPE  *pat;
#endif
	DATATYPE *mean;		/*  4 */
	DATATYPE **rep;		/*  8 */
	int     pruned;		/* 12 */
	int     size;		/* 16 */
	int     root;		/* 20 */
	double   distance;	/* 28 */
	int	full;		/* .. */
	int	*pats;
#ifdef ORIGINAL_VERSION
	int     leaf;
	double   y;		/* not used */
	struct BTree *r_tree, *l_tree;
#endif
};

/* 12 bytes -> 16 records for alignment on 64 bytes boundaries */
typedef struct nnb_info {
	int index;		/* 4 */
	double dist;		/* 12 */
} nnb_info_t;

extern struct BTree *new_tree();

#define LEAF -1

#ifndef TRUE
#define TRUE	1
#endif
#ifndef FALSE
#define FALSE	0
#endif

extern double distance(DATATYPE *pat1, DATATYPE *pat2, int lpat);
extern double my_gettime();
extern double root(double r);
extern double cure_pat_cluster_distance(DATATYPE *p, struct BTree *y, int lpat);
extern void myknn(double **xdata, double *q, int npat, int lpat, int knn, int *nn_x, double *nn_d);
extern double read_nextnum(FILE *fp);
