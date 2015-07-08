//
//  cuda_array_var_int_element.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include <stdio.h>
#include "cuda_array_var_int_element.h"

#if CUDAON

__device__ CudaArrayVarIntElement::CudaArrayVarIntElement ( int n_id, int n_vars, int n_args,
                                                            int* vars, int* args,
                                                            int* domain_idx, uint* domain_states ) :
CudaConstraint ( n_id, n_vars, n_args, vars, args, domain_idx, domain_states ) {
}//CudaArrayVarIntElement

__device__ CudaArrayVarIntElement::~CudaArrayVarIntElement ()
{
}

__device__ void
CudaArrayVarIntElement::consistency ()
{
}//consistency

//! It checks if
__device__ bool
CudaArrayVarIntElement::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
__device__ void
CudaArrayVarIntElement::print() const
{
}//print_semantic
 
#endif


