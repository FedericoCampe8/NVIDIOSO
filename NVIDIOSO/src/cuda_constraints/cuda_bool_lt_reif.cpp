//
//  cuda_bool_lt_reif.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include <stdio.h>
#include "cuda_bool_lt_reif.h"

#if CUDAON

__device__ CudaBoolLtReif::CudaBoolLtReif ( int n_id, int n_vars, int n_args,
                                            int* vars, int* args,
                                            int* domain_idx, uint* domain_states ) :
CudaConstraint ( n_id, n_vars, n_args, vars, args, domain_idx, domain_states ) {
}//CudaBoolLtReif

__device__ CudaBoolLtReif::~CudaBoolLtReif ()
{
}

__device__ void
CudaBoolLtReif::consistency ()
{
}//consistency

//! It checks if
__device__ bool
CudaBoolLtReif::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
__device__ void
CudaBoolLtReif::print() const
{
}//print_semantic
 
#endif


