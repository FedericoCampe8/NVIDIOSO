//
//  cuda_set_in.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include <stdio.h>
#include "cuda_set_in.h"

#if CUDAON

__device__ CudaSetIn::CudaSetIn ( int n_id, int n_vars, int n_args,
                                int* vars, int* args,
                                int* domain_idx, uint* domain_states ) :
CudaConstraint ( n_id, n_vars, n_args, vars, args, domain_idx, domain_states ) {
}//CudaBase

__device__ CudaSetIn::~CudaSetIn ()
{
}

__device__ void
CudaSetIn::consistency ()
{
}//consistency

//! It checks if
__device__ bool
CudaSetIn::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
__device__ void
CudaSetIn::print() const
{
}//print_semantic
 
#endif


