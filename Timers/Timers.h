#ifndef TIMERS_H 
#define TIMERS_H
#include <stdlib.h>

// Definition of a struct to hold pointers to different timers

struct Timers
{
   double *init_timer; // Pointer to track initialization time
   double *find_mdp_timer; // Pointer to track time spent finding MDP
   double *clustering_timer; // Pointer to track time spent during the clustering process
   double *merge_timer; // Pointer to track time spent during the merging phase
   double *update_timer;  // Pointer to track time spent updating cluster data or results
   double *pruning_timer; // Pointer to track time spent in the pruning phase
};    
    
#endif       