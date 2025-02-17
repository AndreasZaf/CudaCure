/*
 * alloc.h -- memory allocation defines for cluster
 *
 */


#define new_2d_array_of(n1, n2, type) \
	(type **)calloc_2d((unsigned)(n1), (unsigned)(n2), sizeof(type))


extern void free_2d_array(char **array, int N);
extern char  **calloc_2d(unsigned int N1, unsigned int N2, unsigned int size);