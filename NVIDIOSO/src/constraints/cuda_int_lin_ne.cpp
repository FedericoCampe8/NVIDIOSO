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
    CudaConstraint ( n_id, n_vars, n_args, vars, args, domain_idx, domain_states )
{
    /*
     * Note: the ids of the variables do not correspond to the ids of the variables
     *       on the host.
     *       These index, are used on the device with a different mapping and they
     *       are consistent with the domain_idx, meaning that domain_idx[ var_x ]
     *       contains the index of the domain for the variable represented by
     *       the id stored in var_x, and so it is for all the other variables.
     */
    _args_size = n_args;
    _as = args;
    _bs = vars;
    _c  = args[ _args_size - 1 ];
    _domain_var = new uint* [ _scope_size ];
    for ( int i = 0; i < _scope_size; i++ )
    {
        _domain_var[ i ] = (domain_states + domain_idx[ vars[i] ]);
    }
}//CudaIntLinNe

__device__ CudaIntLinNe::~CudaIntLinNe () {
    delete [] _domain_var;
}

__device__ void
CudaIntLinNe::consistency ()
{
    /**
     * This function propagates only when there is just
     * variables that is not still assigned.
     * Otherwise it returns without any check.
     */
    // Split consistency according to the number of available threads
    if ( all_ground() )           return;
    if ( !only_one_not_ground() ) return;
	
    // Only one var not ground
    int product = 0;
    int non_ground_idx = -1;
    for ( int idx = 0; idx < _scope_size; idx++ )
    {
        if ( !is_singleton ( _domain_var[ idx ] ) )
        {
            non_ground_idx = idx;
            continue;
        }
        product += _as[ idx ] * min( _domain_var[ idx ] );
    }//var

    // a + kx != c -> x != (c - a) / k
    if ( non_ground_idx != -1 )
    {
    	int avoid = (_c - product) / _as[ non_ground_idx ];
        subtract ( _domain_var[ non_ground_idx ], avoid );
    }
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
    int product = 0;
    
    // ToDo: test if this would be faster if implemented in parallel
    for ( int idx = 0; idx < _scope_size; idx++ )
        product += _as[ idx ] * min ( _domain_var [ idx ] );

    return ( product != _c );
}//satisfied

//! Prints the semantic of this constraint
__device__ void
CudaIntLinNe::print() const {
    if ( threadIdx.x == 0 )
    {
        printf ("c_%d: cuda_int_lin_ne(\n", (int)_unique_id);
        for ( int i = 0; i < _scope_size; i++ )
            printf ("var %d: evt %d, lb %d, ub %d, dsz %d, %u,\n",
                    _bs[ i ], (int)_domain_var[i][0], min(_domain_var[i]),
                    max(_domain_var[i]), (int)_domain_var[i][4], (uint)_domain_var[i][5]
                    );
        printf (")\n");
    }
}//print

#endif
