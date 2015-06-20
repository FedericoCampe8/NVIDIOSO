//
//  cuda_constraint.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 02/12/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "cuda_constraint.h"

#if CUDAON
__device__
CudaConstraint::CudaConstraint ( int n_id, int n_vars, int n_args,
                                 int* vars, int* args,
                                 int* domain_idx, uint* domain_states ) :
_unique_id  ( n_id ),
_scope_size ( n_vars ),
_args_size  ( n_args ),
_vars       ( vars ),
_args       ( args ) {
  _status = (uint**) malloc ( n_vars * sizeof (uint*));
  for ( int i = 0; i < n_vars; i++ ) {
    _status[ i ] = &domain_states[ domain_idx[ _vars[ i ] ] ];
  }//i
}//CudaConstraint

__device__
CudaConstraint::~CudaConstraint () {
  free ( _vars );
  free ( _args );
  free ( _status );
}//~Constraint


__device__ bool 
CudaConstraint::all_ground () const {
	// ToDo: consider whether to implement it in parallel
	for ( int i = 0; i < _scope_size; i++ )
		if ( !is_singleton ( _status[ i ] ) ) return false;
	return true;
}//all_ground

__device__ size_t
CudaConstraint::get_unique_id () const {
  return _unique_id;
}//get_unique_id

__device__ size_t
CudaConstraint::get_scope_size () const {
  return _scope_size;
}//get_scope_size

__device__ size_t
CudaConstraint::get_arguments_size () const {
  return _args_size;
}//get_arguments_size

#endif
