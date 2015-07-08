//
//  cuda_set_subset_reif.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include <stdio.h>
#include "cuda_set_subset_reif.h"

#if CUDAON

__device__ CudaSetSubsetReif::CudaSetSubsetReif ( int n_id, int n_vars, int n_args,
                                                  int* vars, int* args,
                                                  int* domain_idx, uint* domain_states ) :
CudaConstraint ( n_id, n_vars, n_args, vars, args, domain_idx, domain_states ) {
}//CudaSetSubsetReif

__device__ CudaSetSubsetReif::~CudaSetSubsetReif ()
{
}

__device__ void
CudaSetSubsetReif::consistency ()
{
}//consistency

//! It checks if
__device__ bool
CudaSetSubsetReif::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
__device__ void
CudaSetSubsetReif::print() const
{
}//print_semantic
 
#endif


