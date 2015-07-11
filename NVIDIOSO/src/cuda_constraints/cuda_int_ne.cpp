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
}//CudaIntNe

__device__ CudaIntNe::~CudaIntNe () {
}

__device__ void
CudaIntNe::consistency ()
{ 
    /*
     * Propagate constraint iff there are two
     * FD variables and one is ground OR
     * there is one FD variable which is not ground.
     * @note (2 - _scope_size) can be used instead of
     *       get_arguments_size function since
     *       (2 - _scope_size) = get_arguments_size ().
     */
    if ( NUM_ARGS == 2 ) return;

    if ( all_ground () ) return;
    
    // 1 FD variable: if not singleton, propagate.
    if ( NUM_ARGS == 1 )
    {
    	if ( !is_singleton ( X_VAR ) )
        {
            // 1 Auxiliary argument here
            subtract( X_VAR, ARGS [ 0 ] );
        }
    	return;
    }
   
    /* 
     * 2 FD variables: if one is singleton,
     * propagate on the other.
     */
    if ( NUM_VARS == 2 )
    {
    	bool singleton_x = is_singleton ( X_VAR );
    	bool singleton_y = is_singleton ( Y_VAR );
        if ( singleton_x && !singleton_y ) 
        {
            subtract ( Y_VAR, get_min ( X_VAR ) );
        }
        else if ( !singleton_x && singleton_y ) 
		{
            subtract ( X_VAR, get_min ( Y_VAR ) );
        }
        return;
    }
}//consistency

//! It checks if x != y
__device__ bool
CudaIntNe::satisfied ()
{
    // No FD variables, just check the integers values
    if ( NUM_ARGS == 2 ) 
    {
        return ARGS[ X_VAR ] != ARGS[ Y_VAR ];
    }
  
    // 1 FD variable, if singleton check
    if ( (NUM_ARGS == 1) && is_singleton ( X_VAR ) )
    {
        if ( ARGS [ 0 ] != get_min ( X_VAR ) )
        {
        	return true;
        }
        GET_VAR_EVT(X_VAR) = FAL_EVT;
        return false;
    }
  
    // 2 FD variables, if singleton check
    if ( is_singleton ( X_VAR ) && is_singleton ( Y_VAR ) ) 
    {
        if ( get_min ( X_VAR ) != get_min ( Y_VAR ) )
        {
        	return true;
        }
        GET_VAR_EVT(X_VAR) = FAL_EVT;
        GET_VAR_EVT(Y_VAR) = FAL_EVT;
        return false;
    }
  
    /*
     * Check if a domain is empty.
     * If it is the case: failed propagation.
     */
    if ( is_empty ( X_VAR ) || is_empty ( Y_VAR ) )
    {
    	GET_VAR_EVT(X_VAR) = FAL_EVT;
        GET_VAR_EVT(Y_VAR) = FAL_EVT;
        return false;
    }
    
    /*
     * Other cases: there is not enough information
     * to state whether the constraint is satisfied or not.
     * Return true.
     */
    return true;
}//satisfied

//! Prints the semantic of this constraint
__device__ void
CudaIntNe::print() const
{
    if ( threadIdx.x != 0 )
        return;
    
    if ( NUM_ARGS == 2 )
    {
        printf ( "c_%d: int_ne(int: %d, int: %d)\n",
                 (int)_unique_id,
                 ARGS [ X_VAR ],
                 ARGS [ Y_VAR ] );
    }
    else if ( NUM_ARGS == 1 )
    {
        printf ( "c_%d: int_ne(var int: %d, int: %d)\n",
                 (int)_unique_id,
                 VARS [ X_VAR ],
                 C_ARG );
    }
    else
    {
        printf ( "c_%d: int_ne(var int: %d [%d, %d](%d), var int: %d [%d, %d](%d))\n",
                 (int)_unique_id,
                 VARS [ X_VAR ], GET_VAR_LB(X_VAR), GET_VAR_UB(X_VAR), GET_VAR_DSZ(X_VAR),
                 VARS [ Y_VAR ], GET_VAR_LB(Y_VAR), GET_VAR_UB(Y_VAR), GET_VAR_DSZ(Y_VAR) );
    }
}//print_semantic
 
#endif


