//
//  cuda_array_int_element.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include <stdio.h>
#include "cuda_array_int_element.h"

#if CUDAON

__device__ CudaArrayIntElement::CudaArrayIntElement ( int n_id, int n_vars, int n_args,
                                                      int* vars, int* args,
                                                      int* domain_idx, uint* domain_states ) :
CudaConstraint ( n_id, n_vars, n_args, vars, args, domain_idx, domain_states ) {
}//CudaArrayIntElement

__device__ CudaArrayIntElement::~CudaArrayIntElement ()
{
}

__device__ void
CudaArrayIntElement::consistency ()
{
}//consistency

//! It checks if
__device__ bool
CudaArrayIntElement::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
__device__ void
CudaArrayIntElement::print() const
{
}//print_semantic
 
#endif


