//
//  cuda_bool_le_reif.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include <stdio.h>
#include "cuda_bool_le_reif.h"

#if CUDAON

__device__ CudaBoolLeReif::CudaBoolLeReif ( int n_id, int n_vars, int n_args,
                                            int* vars, int* args,
                                            int* domain_idx, uint* domain_states ) :
CudaConstraint ( n_id, n_vars, n_args, vars, args, domain_idx, domain_states ) {
}//CudaBoolLeReif

__device__ CudaBoolLeReif::~CudaBoolLeReif ()
{
}

__device__ void
CudaBoolLeReif::consistency ()
{
}//consistency

//! It checks if
__device__ bool
CudaBoolLeReif::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
__device__ void
CudaBoolLeReif::print() const
{
}//print_semantic
 
#endif


