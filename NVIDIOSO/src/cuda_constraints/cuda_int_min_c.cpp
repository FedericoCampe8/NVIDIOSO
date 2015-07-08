//
//  cuda_int_min_c.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include <stdio.h>
#include "cuda_int_min_c.h"

#if CUDAON

__device__ CudaIntMinC::CudaIntMinC ( int n_id, int n_vars, int n_args,
                                      int* vars, int* args,
                                      int* domain_idx, uint* domain_states ) :
CudaConstraint ( n_id, n_vars, n_args, vars, args, domain_idx, domain_states ) {
}//CudaIntMinC

__device__ CudaIntMinC::~CudaIntMinC ()
{
}

__device__ void
CudaIntMinC::consistency ()
{
}//consistency

//! It checks if
__device__ bool
CudaIntMinC::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
__device__ void
CudaIntMinC::print() const
{
}//print_semantic
 
#endif


