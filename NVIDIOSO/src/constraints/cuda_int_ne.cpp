//
//  cuda_int_ne.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 03/12/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include <stdio.h>
#include "cuda_int_ne.h"

#if CUDAON

__device__ CudaIntNe::CudaIntNe ( int n_id, int n_vars, int n_args,
                                  int* vars, int* args,
                                  int* domain_idx, uint* domain_states ) :
CudaConstraint ( n_id, n_vars, n_args, vars, args, domain_idx, domain_states ) {
	/*
	 * Note: the ids of the variables do not correspond to the ids of the variables
	 * 		 on the host. 
	 *       These index, are used on the device with a different mapping and they
	 *       are consistent with the domain_idx, meaning that domain_idx[ _var_x ]
	 *       contains the index of the domain for the variable represented by 
	 *       the id stored in _var_x, and so it is for _var_y.
	 * Note: to save space on device we store both vars idx and arguments in the same
	 *       variables (i.e., _var_x, _var_y).
	 *       If n_args == 1, _var_y will always contain the int value.
	 */
	_n_arg = n_args;
	if ( _n_arg == 0 ) 
	{
		_var_x = vars[ 0 ];
		_var_y = vars[ 1 ];
		
		_domain_x = &domain_states[ domain_idx[ _var_x ] ];
		_domain_y = &domain_states[ domain_idx[ _var_y ] ];
	}
	else if ( _n_arg == 1 ) 
	{
		_var_x = vars[ 0 ];
		_var_y = args[ 0 ];
		
		_domain_x = &domain_states[ domain_idx[ _var_x ] ];
	}
	else {
		_var_x = args[ 0 ];
		_var_y = args[ 1 ];
	}
}//CudaIntNe

__device__ CudaIntNe::~CudaIntNe () {
}

__device__ void
CudaIntNe::consistency () { 
 
  /*
   * Propagate constraint iff there are two
   * FD variables and one is ground OR
   * there is one FD variable which is not ground.
   * @note (2 - _scope_size) can be used instead of
   *       get_arguments_size function since
   *       (2 - _scope_size) = get_arguments_size ().
   */
  if ( _n_arg == 2 ) return;
  
  // 1 FD variable: if not singleton, propagate.
	if ( _n_arg == 1 )
	{
    	if ( !is_singleton ( _domain_x ) )
      		subtract( _domain_x, _var_y );
    	return;
	}
   
  /* 
   * 2 FD variables: if one is singleton,
   * propagate on the other.
   */
	if ( _n_arg == 0 ) {
    if ( is_singleton ( _domain_x ) && !is_singleton ( _domain_y ) ) 
    {
    	subtract ( _domain_y, _domain_x[ 2 ] );
    }
    else if ( !is_singleton ( _domain_x ) && is_singleton ( _domain_y ) ) 
	{
		subtract ( _domain_x, _domain_y[ 2 ] );
    }
    return;
  }
}//consistency

//! It checks if x != y
__device__ bool
CudaIntNe::satisfied ()  {

  // No FD variables, just check the integers values
  if ( _n_arg == 2 ) 
    return _var_x != _var_y;
  
  // 1 FD variable, if singleton check
  if ( (_n_arg == 1) &&
  		is_singleton ( _domain_x ) ) 
	{
    return _var_y != min ( _domain_x );
  	}
  
  // 2 FD variables, if singleton check
  if (	is_singleton ( _domain_x ) &&
		is_singleton ( _domain_y ) ) 
	{
    return min ( _domain_x ) != min ( _domain_y );
  	}
  
  /*
   * Check if a domain is empty.
   * If it is the case: failed propagation.
   */
   if ( _domain_x[ 0 ] == 6 || _domain_y[ 0 ] == 6 ) return false;
   
  /*
   * Other cases: there is not enough information
   * to state whether the constraint is satisfied or not.
   * Return true.
   */
  return true;
}//satisfied

//! Prints the semantic of this constraint
__device__ void
CudaIntNe::print() const {
  //printf ( "c_%d: int_ne(var int: %d, var int: %d)\n",  (int)_unique_id, _var_x, _var_y );
  printf ("c_%d: int_ne(var %d: evt %d, lb %d, ub %d, dsz %d, %u,  var %d: evt %d, lb %d, ub %d, dsz %d, %u)\n",
  	(int)_unique_id, 
  	_var_x, 
	(int)_domain_x[0], min (_domain_x), max (_domain_x), (int)_domain_x[4], (uint)_domain_x[5],
	_var_y,
	(int)_domain_y[0], min (_domain_y), max (_domain_y), (int)_domain_y[4], (uint)_domain_y[5]);
}//print_semantic
 
#endif


