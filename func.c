#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

FILE *open_traindata()
{
	FILE *fp;

	fp = fopen("traindata.txt", "r");
	if (fp == NULL) {
		printf("File not available\n");
		exit(1);
	}
	return fp;
}

FILE *open_querydata()
{
	FILE *fp;

	fp = fopen("querydata.txt", "r");
	if (fp == NULL) {
		printf("File not available\n");
		exit(1);
	}
	return fp;
}

void fitfun(double *x, int n, double *res)
{
	int i;
	double f = 0.0;

	for (i=0; i<n-1; i++)   /* rosenbrock */
		f = f + 100.0*pow((x[i+1]-x[i]*x[i]),2) + pow((x[i]-1.0),2);

	*res = f;
	return;
}

double cross(double *x, int n) {
	double res;
	int i;

	fitfun(x, n, &res);
	return res;
}


#if 1
void print_matrix(char *title, double *v, int n)
{
	int i;

//	if (!display) return;

	printf("\n%s =\n\n", title);
	for (i = 0; i < n; i++) {
		printf("   %20.15lf\n", v[i]);
	}
	printf("\n");
}

void print_matrix_2d(char *title, double **v, int n1, int n2)
{
	int i, j;

//	if (!display) return;

	printf("\n%s =\n\n", title);
	for (i = 0; i < n1; i++) {
		for (j = 0; j < n2; j++) {
			printf("   %20.15lf", v[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

double compute_min(double *v, int n)
{
	int i;
	double vmin = v[0];
	for (i = 1; i < n; i++)
		if (v[i] < vmin) vmin = v[i];

	return vmin;
}

double compute_max(double *v, int n)
{
	int i;
	double vmax = v[0];
	for (i = 1; i < n; i++)
		if (v[i] > vmax) vmax = v[i];

	return vmax;
}

double compute_sum(double *v, int n)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += v[i];

	return s;
}

double compute_sum_pow(double *v, int n, int p)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += pow(v[i], p);

	return s;
}

double compute_mean(double *v, int n)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += v[i];

	return s/n;
}

double compute_std(double *v, int n, double mean)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += pow(v[i]-mean,2);

	return sqrt(s/(n-1));
}

double compute_var(double *v, int n, double mean)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += pow(v[i]-mean,2);

	return s/n;
}

double compute_dist(double *v, double *w, int n)
{
	int i;
	double s = 0.0, df;
	for (i = 0; i < n; i++) {
		//df = v[i]-w[i];
		//s += df*df;
		s+= pow(v[i]-w[i],2);
	}

	return sqrt(s);
}

double compute_dist_v(double *v, double *w, int n)
{
	int i;
	double s = 0.0, df;
	printf("compute_dist(%p,%p)\n", v, w);
	for (i = 0; i < n; i++) {
		printf("%d : v=%.5lf w=%.5lf\n", i, v[i], w[i]);
		df = v[i]-w[i];
		s += df*df;
		// s+= pow(v[i]-w[i],2);
	}

	return sqrt(s);
}

double compute_max_pos(double *v, int n, int *pos)
{
	int i, p = 0;
	double vmax = v[0];
	for (i = 1; i < n; i++)
		if (v[i] > vmax) {
			vmax = v[i];
			p = i;
		}

	*pos = p;
	return vmax;
}

double compute_min_pos(double *v, int n, int *pos)
{
	int i, p = 0;
	double vmin = v[0];
	for (i = 1; i < n; i++)
		if (v[i] < vmin) {
			vmin = v[i];
			p = i;
		}

	*pos = p;
	return vmin;
}

double compute_root(double dist, int norm)
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

double compute_distance(double *pat1, double *pat2, int lpat, int norm)
{
	register int i;
	double dist = 0.0;

	for (i = 0; i < lpat; i++) {
		double diff = 0.0;

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

	return dist;	// compute_root(dist);
}

void compute_knn_bfs(double **xdata, double *q, int npat, int lpat, int knn, int *nn_x, double *nn_d)
{
	int i, max_i;
	double max_d, new_d;
	int norm = 2;

	for (i = 0; i < knn; i++) {
		nn_x[i] = -1;
		nn_d[i] = 1e99-i;
	}

	max_d = compute_max_pos(nn_d, knn, &max_i);
	for (i = 0; i < npat; i++) {
		new_d = compute_dist(q, xdata[i], lpat);	// +norm
		if (new_d < max_d) {
			nn_x[max_i] = i;
			nn_d[max_i] = new_d;
		}
		max_d = compute_max_pos(nn_d, knn, &max_i);
	}

	int temp_x, j;
	double temp_d;
	for (i = (knn - 1); i > 0; i--) {
		for (j = 1; j <= i; j++) {
			if (nn_d[j-1] > nn_d[j]) {
				temp_d = nn_d[j-1]; nn_d[j-1] = nn_d[j]; nn_d[j] = temp_d;
				temp_x = nn_x[j-1]; nn_x[j-1] = nn_x[j]; nn_x[j] = temp_x;
			}
		}
	}

	return;
}
#endif

#ifndef MAX_NNB
#define MAX_NNB	256
#endif

double predict_value(int dim, int knn, double *xdata, double *ydata, double *point, double *dist)
{
	int i, j;
	double sum_v = 0.0;
#if 1	// plain mean
	for (i = 0; i < knn; i++) {
		sum_v += ydata[i];
	}

	return sum_v/knn;
#endif
#if 0	// IDW (inverse distance weight)
	double nn_w[MAX_NNB];
	double tot_d;
	double idw_p = 1.0;

	tot_d = compute_sum_pow(dist, knn, idw_p);
	for (i = 0; i < knn; i++) {
		nn_w[i] = pow(dist[i],idw_p)/tot_d;
	}

	for (i = 0; i < knn; i++) {
		sum_v += nn_w[i]*ydata[i];
	}
	return sum_v;
#endif
}