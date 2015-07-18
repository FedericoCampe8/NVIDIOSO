//
//  cuda_simple_constraint_store_seq.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/18/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//


#include "cuda_simple_constraint_store_1b1c.h"
#include "cuda_propagation_utilities.h"
#include "cuda_synchronization_utilities.h"

using namespace std;

CudaSimpleConstraintStore1b1c::CudaSimpleConstraintStore1b1c () :
    CudaSimpleConstraintStore () {
    _dbg = "CudaSimpleConstraintStore1b1c - ";
}//CudaSimpleConstraintStore

CudaSimpleConstraintStore1b1c::~CudaSimpleConstraintStore1b1c () {
}//~CudaSimpleConstraintStore1b1c

void
CudaSimpleConstraintStore1b1c::dev_consistency ()
{

#if CUDAON

	CudaPropUtils::cuda_consistency_1b1c <<< _constraint_queue.size(), 1, _scope_state_size >>> ( _d_constraint_queue );
    CudaSynchUtils::cuda_set_domains_from_bit_1b1v<<< _cp_model_ptr->num_variables (), 1, STANDARD_DOM >>> 
    ( 
		_cp_model_ptr->get_dev_domain_index_ptr  (),
    	_cp_model_ptr->get_dev_domain_states_ptr () 
	);
	
#endif

}//dev_consistency



