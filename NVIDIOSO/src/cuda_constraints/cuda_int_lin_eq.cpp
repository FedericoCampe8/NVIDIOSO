//
//  cuda_int_lin_eq.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include <stdio.h>
#include "cuda_int_lin_eq.h"

#if CUDAON

__device__ CudaIntLinEq::CudaIntLinEq ( int n_id, int n_vars, int n_args,
                                        int* vars, int* args,
                                        int* domain_idx, uint* domain_states ) :
CudaConstraint ( n_id, n_vars, n_args, vars, args, domain_idx, domain_states ) {
}//CudaIntLinEq

__device__ CudaIntLinEq::~CudaIntLinEq ()
{
}

__device__ void
CudaIntLinEq::consistency ( int ref )
{
}//consistency

//! It checks if
__device__ bool
CudaIntLinEq::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
__device__ void
CudaIntLinEq::print() const
{
}//print_semantic
 
#endif


