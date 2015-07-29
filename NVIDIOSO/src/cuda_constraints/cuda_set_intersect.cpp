//
//  cuda_set_intersect.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include <stdio.h>
#include "cuda_set_intersect.h"

#if CUDAON

__device__ CudaSetIntersect::CudaSetIntersect ( int n_id, int n_vars, int n_args,
                                                int* vars, int* args,
                                                int* domain_idx, uint* domain_states ) :
CudaConstraint ( n_id, n_vars, n_args, vars, args, domain_idx, domain_states ) {
}//CudaSetIntersect

__device__ CudaSetIntersect::~CudaSetIntersect ()
{
}//CudaSetIntersect

__device__ void
CudaSetIntersect::consistency ( int ref )
{
}//consistency

//! It checks if
__device__ bool
CudaSetIntersect::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
__device__ void
CudaSetIntersect::print() const
{
}//print_semantic
 
#endif


