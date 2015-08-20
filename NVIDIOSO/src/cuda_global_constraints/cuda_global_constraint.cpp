//
//  cuda_constraint.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 19/08/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "cuda_global_constraint.h"

#if CUDAON

__device__
CudaGlobalConstraint::CudaGlobalConstraint ( int n_id, int n_vars, int n_args,
                                 			 int* vars, int* args,
                                 			 int* domain_idx, uint* domain_states,
                                 			 int num_blocks, int num_threads ) :
	CudaConstraint    ( n_id, n_vars, n_args, vars, args, domain_idx, domain_states ),
	_num_blocks       ( num_blocks ),
	_num_threads      ( num_threads ),
	_consistency_type ( 0 ) {
}//CudaGlobalConstraint

__device__
CudaGlobalConstraint::~CudaGlobalConstraint () {
}//~Constraint

__device__ void 
CudaGlobalConstraint::set_consistency_type ( int con_type )
{
	_consistency_type = con_type;
}//set_consistency_type

__device__ void 
CudaGlobalConstraint::consistency ( int ref )
{
	if ( _consistency_type == 0 ) 
	{
		naive_consistency ();
	}
	else if ( _consistency_type == 1 ) 
	{
		bound_consistency ();	
	}
	else if ( _consistency_type == 2 ) 
	{
		full_consistency ();
	}
	else
	{
		naive_consistency ();
	}
}//consistency

__device__ bool 
CudaGlobalConstraint::satisfied ()
{
	return true;
}//satisfied

__device__ void 
CudaGlobalConstraint::print () const
{
}//print

#endif
