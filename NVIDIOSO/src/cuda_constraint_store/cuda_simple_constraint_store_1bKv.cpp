//
//  cuda_simple_constraint_store_1bKv.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/21/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//


#include "cuda_simple_constraint_store_1bKv.h"
#include "cuda_propagation_utilities.h"
#include "cuda_synchronization_utilities.h"

using namespace std;

CudaSimpleConstraintStore1bKv::CudaSimpleConstraintStore1bKv () {
    _dbg = "CudaSimpleConstraintStore1bKv - ";
}//CudaSimpleConstraintStore1b1v

CudaSimpleConstraintStore1bKv::~CudaSimpleConstraintStore1bKv () {
}//~CudaSimpleConstraintStore1bKv

void
CudaSimpleConstraintStore1bKv::move_queue_to_device ()
{
    CudaSimpleConstraintStore1b1v::move_queue_to_device ();

#if CUDAON
  
  	int total_blocks = dev_grid_size.x;
    int vars_within_block = 1;
    _shared_memory = _scope_state_size;
    while ( _shared_memory < _shared_limit && total_blocks >= 1 && (vars_within_block * WARP_SIZE ) < _max_block_size )
    {	
    	++vars_within_block;
    	_shared_memory += _scope_state_size;
    	total_blocks = dev_grid_size.x / vars_within_block;
    }
    _shared_memory -= _scope_state_size;
    _shared_memory = max ( _shared_memory, _scope_state_size );
    
	dev_grid_size.x   = total_blocks + 1;
	dev_block_size.x  = max ( (vars_within_block - 1) * WARP_SIZE, 1 );
	 
#endif

}//move_queue_to_device

void
CudaSimpleConstraintStore1bKv::dev_consistency ()
{

#if CUDAON

    if ( _shared_memory > _shared_limit )
    {
        sequential_propagation ();
    }
    else
    {
        // Propagate constraints in parallel
        if ( _h_constraint_queue_idx.size() == 1 ) return;
        CudaPropUtils::cuda_consistency_1bKv <<< dev_grid_size, dev_block_size, _shared_memory >>> 
        ( _d_constraint_queue, _d_constraint_queue_idx, 
         _h_constraint_queue_idx.size() - 1, _scope_state_size / sizeof(uint), 
         STANDARD_DOM, _d_states_aux );
        
        // Reset mappings for next propagation
        reset ();
         
        dev_grid_size.x  = _cp_model_ptr->num_variables ();
        dev_block_size.x = 1;
        CudaSynchUtils::cuda_set_domains_from_bit_1b1v <<< dev_grid_size, dev_block_size, STANDARD_DOM * sizeof ( uint ) >>>
        ( _cp_model_ptr->get_dev_domain_index_ptr (), _d_states_aux );
    }

#endif

}//dev_consistency



