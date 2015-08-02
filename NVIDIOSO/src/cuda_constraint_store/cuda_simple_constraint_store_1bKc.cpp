
//
//  cuda_simple_constraint_store_1bKc.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/02/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//


#include "cuda_simple_constraint_store_1bKc.h"
#include "cuda_propagation_utilities.h"
#include "cuda_synchronization_utilities.h"

using namespace std;

CudaSimpleConstraintStore1bKc::CudaSimpleConstraintStore1bKc () :
	_grid_size  	 		   ( 1 ),
	_block_size 	 		   ( 1 ),
	_shared_mem_size 		   ( 0 ), 
	_num_constraints_per_block ( 1 ) {
    _dbg = "CudaSimpleConstraintStore1bKc - ";
}//CudaSimpleConstraintStore1b1c

CudaSimpleConstraintStore1bKc::~CudaSimpleConstraintStore1bKc () {
}//~CudaSimpleConstraintStore1bKc

void 
CudaSimpleConstraintStore1bKc::init_store ()
{
	CudaSimpleConstraintStore1b1c::init_store ();
	
	/*
	 * Get the max number of threads per block
	 * given the amount of shared memory to allocate ans the size of the scope.
	 */
	 
	_num_constraints_per_block = _max_block_size / WARP_SIZE;
	_shared_mem_size           = _num_constraints_per_block * _scope_state_size;
	while ( _shared_mem_size > _shared_limit && _num_constraints_per_block > 0 )
	{
		_num_constraints_per_block--;
		_shared_mem_size = _num_constraints_per_block * _scope_state_size;
	}	
	
	if ( _num_constraints_per_block == 0 ) return;
	
	/*
	 * num_constraints_per_block now contains 
	 * the number of constraints to propagate per block.
	 */
	_block_size = _num_constraints_per_block * WARP_SIZE;
}//init_store

void
CudaSimpleConstraintStore1bKc::dev_consistency ()
{
	
#if CUDAON

	_grid_size = (size_t) ceil ( _constraint_queue.size() / (_num_constraints_per_block * 1.0) );
    CudaPropUtils::cuda_consistency_1bKc <<< _grid_size, _block_size, _shared_mem_size >>>
    ( _d_constraint_queue, _constraint_queue.size () );

    // Calculate the set of variables modified by the previous propagation
    CudaSimpleConstraintStore1b1c::set_vars_to_update_after_prop ();
	
    /*
     * Synchronize domains and make them consistent
     * @todo synchronize only the variable that have effectively changed
     */
    CudaSynchUtils::cuda_set_domains_from_bit_1b1v <<< dev_grid_size, dev_block_size, STANDARD_DOM * sizeof ( uint ) >>>
    ( _domains_idx_ptr, _domains_ptr );
    
#endif

}//dev_consistency



