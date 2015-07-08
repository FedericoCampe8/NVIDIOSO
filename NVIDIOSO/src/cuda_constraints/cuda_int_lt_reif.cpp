//
//  cuda_int_lt_reif.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include <stdio.h>
#include "cuda_int_lt_reif.h"

#if CUDAON

__device__ CudaIntLtReif::CudaIntLtReif ( int n_id, int n_vars, int n_args,
                                          int* vars, int* args,
                                          int* domain_idx, uint* domain_states ) :
CudaConstraint ( n_id, n_vars, n_args, vars, args, domain_idx, domain_states ) {
}//CudaIntLtReif

__device__ CudaIntLtReif::~CudaIntLtReif ()
{
}

__device__ void
CudaIntLtReif::consistency ()
{
}//consistency

//! It checks if
__device__ bool
CudaIntLtReif::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
__device__ void
CudaIntLtReif::print() const
{
}//print_semantic
 
#endif


