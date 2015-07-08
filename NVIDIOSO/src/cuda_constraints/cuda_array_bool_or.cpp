//
//  cuda_array_bool_or.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include <stdio.h>
#include "cuda_array_bool_or.h"

#if CUDAON

__device__ CudaArrayBoolOr::CudaArrayBoolOr ( int n_id, int n_vars, int n_args,
                                       int* vars, int* args,
                                       int* domain_idx, uint* domain_states ) :
CudaConstraint ( n_id, n_vars, n_args, vars, args, domain_idx, domain_states ) {
}//CudaArrayBoolOr

__device__ CudaArrayBoolOr::~CudaArrayBoolOr ()
{
}

__device__ void
CudaArrayBoolOr::consistency ()
{
}//consistency

//! It checks if
__device__ bool
CudaArrayBoolOr::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
__device__ void
CudaArrayBoolOr::print() const
{
}//print_semantic
 
#endif


