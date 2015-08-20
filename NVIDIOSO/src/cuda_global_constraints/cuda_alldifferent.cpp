//
//  cuda_alldifferent.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 19/08/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include <stdio.h>
#include "cuda_alldifferent.h"

#if CUDAON

__device__ CudaAlldifferent::CudaAlldifferent ( int n_id, int n_vars, int n_args,
                                  				int* vars, int* args,
                                  				int* domain_idx, uint* domain_states,
                                  				int num_blocks, int num_threads ) :
	CudaGlobalConstraint ( n_id, n_vars, n_args, vars, args, domain_idx, domain_states, num_blocks, num_threads ) {
}//CudaAlldifferent

__device__ CudaAlldifferent::~CudaAlldifferent () {
}//~CudaAlldifferent

__device__ void 
CudaAlldifferent::naive_consistency ()
{
}//naive_consistency

__device__ void 
CudaAlldifferent::bound_consistency ()
{
}//bound_consistency

__device__ void 
CudaAlldifferent::full_consistency ()
{
}//full_consistency

__device__ bool 
CudaAlldifferent::satisfied ()
{
	return true;
}//satisfied

__device__ void 
CudaAlldifferent::print () const
{
}//print

#endif


