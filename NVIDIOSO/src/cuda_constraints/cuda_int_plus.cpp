//
//  cuda_int_plus.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include <stdio.h>
#include "cuda_int_plus.h"

#if CUDAON

__device__ CudaIntPlus::CudaIntPlus ( int n_id, int n_vars, int n_args,
                                      int* vars, int* args,
                                      int* domain_idx, uint* domain_states ) :
CudaConstraint ( n_id, n_vars, n_args, vars, args, domain_idx, domain_states ) {
}//CudaIntPlus

__device__ CudaIntPlus::~CudaIntPlus ()
{
}

__device__ void
CudaIntPlus::consistency ()
{
}//consistency

//! It checks if
__device__ bool
CudaIntPlus::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
__device__ void
CudaIntPlus::print() const
{
}//print_semantic
 
#endif


