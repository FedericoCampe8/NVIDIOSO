//
//  cuda_bool_2_int.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include <stdio.h>
#include "cuda_bool_2_int.h"

#if CUDAON

__device__ CudaBool2Int::CudaBool2Int ( int n_id, int n_vars, int n_args,
                                        int* vars, int* args,
                                        int* domain_idx, uint* domain_states ) :
CudaConstraint ( n_id, n_vars, n_args, vars, args, domain_idx, domain_states ) {
}//CudaBool2Int

__device__ CudaBool2Int::~CudaBool2Int ()
{
}

__device__ void
CudaBool2Int::consistency ()
{
}//consistency

//! It checks if
__device__ bool
CudaBool2Int::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
__device__ void
CudaBool2Int::print() const
{
}//print_semantic
 
#endif


