/*
 * alloc.c -- Memory allocation utilities for cluster
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "alloc.h"

void free_2d_array(char**array,int N)
{
	int i;
	for (i = 0; i<N; i++)
		free(array[i]);
	free((char *) array);
}

char **calloc_2d(unsigned int N1, unsigned int N2, unsigned int size)
{
	int i;
	char **ptr = (char **) calloc(N1, sizeof(char *));
	if (ptr == NULL) {
		return NULL;
	}
	for (i = 0; i < N1; i++)
		if (NULL == (ptr[i] = calloc(N2, size)))
			return NULL;
	return ptr;
}