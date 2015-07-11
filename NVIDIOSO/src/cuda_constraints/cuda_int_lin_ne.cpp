//
//  cuda_int_lin_ne.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 11/06/15.
//  Copyright (c) 2015 ___UDNMSU___. All rights reserved.
//

#include <stdio.h>
#include "cuda_int_lin_ne.h"

#if CUDAON

__device__ CudaIntLinNe::CudaIntLinNe ( int n_id, int n_vars, int n_args,
                                        int* vars, int* args,
                                        int* domain_idx, uint* domain_states ) :
    CudaConstraint ( n_id, n_vars, n_args, vars, args, domain_idx, domain_states ) {
}//CudaIntLinNe

__device__ CudaIntLinNe::~CudaIntLinNe ()
{
}

__device__ void
CudaIntLinNe::consistency ()
{return;
    /**
     * This function propagates only when there is just
     * variables that is not still assigned.
     * Otherwise it returns without any check.
     */
    if ( all_ground() )           return;

    if ( !only_one_not_ground() ) return;

    int idx_not_singleton = get_not_ground ();
    int prod_ground       = get_sum_ground ();

    // a + kx != c -> x != (c - a) / k
    int no_good_element = ( ARGS [ LAST_ARG_IDX ] - prod_ground ) / ARGS [ idx_not_singleton ];
    subtract ( idx_not_singleton, no_good_element );
}//consistency

//! It checks whether the constraint is satisfied
__device__ bool
CudaIntLinNe::satisfied () 
{
    /*
     * If not variables are ground, then there
     * is not enough information to state whether the constraint
     * is satisfied or not.
     * Return true.
     */
    if ( !all_ground() ) return true;
    int product = get_sum_ground ();
    
    if ( product != ARGS [ LAST_ARG_IDX ] )
    {
    	return true;
    }
    GET_VAR_EVT(X_VAR) = FAL_EVT;
    //GET_VAR_EVT(Y_VAR) = FAL_EVT;
    return false;
}//satisfied

//! Prints the semantic of this constraint
__device__ void
CudaIntLinNe::print() const
{
    if ( threadIdx.x != 0 )
        return;
        
    printf ("c_%d: cuda_int_lin_ne != %d\n",
            ((int)_unique_id), ARGS [ LAST_ARG_IDX ] );
}//print

#endif
