//
//  cuda_simple_constraint_store.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 04/12/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//


#include "cuda_simple_constraint_store.h"
#include "cuda_propagation_utilities.h"
#include "cuda_synchronization_utilities.h"

using namespace std;

CudaSimpleConstraintStore::CudaSimpleConstraintStore () :
    SimpleConstraintStore (),
    _d_constraint_queue   ( nullptr ),
    _cp_model_ptr         ( nullptr ) {
    _dbg = "CudaSimpleConstraintStore - ";
}//CudaSimpleConstraintStore

CudaSimpleConstraintStore::~CudaSimpleConstraintStore () {
#if CUDAON
    cudaFree ( _d_constraint_queue );
#endif
}//~CudaSimpleConstraintStore

void
CudaSimpleConstraintStore::finalize ( CudaCPModel* ptr )
{
    // Sanity check
    assert ( ptr != nullptr );

    LogMsg << _dbg + "finalize: set CudaCPModel and allocate queue on device" << std::endl;
    _cp_model_ptr = ptr;

    #if CUDAON
    // Synchronization barrier for blocks on device
    if ( logger.cuda_handle_error ( cudaMalloc ( (void**)&g_dev_synch_barrier,
                                                 sizeof ( bool )) ) )
    {
        string err = _dbg + "finalize: Bad memory allocation on device.\n";
        throw NvdException ( err.c_str(), __FILE__, __LINE__ );
    }

    // Constraint queue on device
    size_t num_constraints = _cp_model_ptr->num_constraints ();
    if ( logger.cuda_handle_error ( cudaMalloc ( (void**)&_d_constraint_queue,
                                                 num_constraints * sizeof ( size_t )) ) )
    {
        string err = _dbg + "finalize: Bad memory allocation on device.\n";
        throw NvdException ( err.c_str(), __FILE__, __LINE__ );
    }
    _h_constraint_queue.insert ( _h_constraint_queue.end(), num_constraints, 0 );
    #endif
}//finalize


bool
CudaSimpleConstraintStore::move_states_to_device () 
{
  return _cp_model_ptr->upload_device_state ();
}//move_states_to_device

bool
CudaSimpleConstraintStore::move_states_from_device () 
{
  return _cp_model_ptr->download_device_state ();
}//move_states_to_device

void
CudaSimpleConstraintStore::move_queue_to_device ()
{
#if CUDAON

    _scope_state_size = 0;
    int i = 0, scope_size = 0;
    for ( auto c: _constraint_queue )
    {
        scope_size = ( _lookup_table.find( c )->second->get_scope_size() * STANDARD_DOM * sizeof (int) );
        if ( _scope_state_size < scope_size )
        {
            _scope_state_size = scope_size;
        }
        _h_constraint_queue[ i++ ] = _cp_model_ptr->constraint_mapping_h_d[ c ];
    }

    cudaMemcpy( _d_constraint_queue, &_h_constraint_queue[0],
                _constraint_queue.size() * sizeof( size_t ), cudaMemcpyHostToDevice );
#endif
}//move_queue_to_device

bool
CudaSimpleConstraintStore::consistency ()
{
#if CUDAON
    
    if ( _d_constraint_queue == nullptr )
    {
        string err = _dbg + "CudaSimpleConstraintStore - Memory on device not allocated.\n";
        throw NvdException ( err.c_str(), __FILE__, __LINE__ );
    }

    // Check for some failure happened somewhere else
    if ( _failure )
    {
        handle_failure ();
        return false;
    }

    // Update states on device
    move_states_to_device ();

    // Move queue on device
    move_queue_to_device ();

    // Propagate constraints on device
    dev_consistency ();

    // Clear queue since it is not needed anymore
    clear_queue ();

    // Update states on host
    bool success = move_states_from_device ();

    if ( !success )
    {// Fail propagation
        clear_queue ();
        return false;
    }
    
#endif

    return true;
}//consistency

void
CudaSimpleConstraintStore::dev_consistency ()
{
#if CUDAON

	if ( solver_params != nullptr )
	{
		switch ( solver_params->cstore_get_dev_propagation () )
		{
			case CudaPropParam::SEQUENTIAL :
				CudaPropUtils::cuda_consistency_sequential <<< 1, 1, _scope_state_size >>> ( _d_constraint_queue, _constraint_queue.size() );
				break;
			case CudaPropParam::BLOCK_PER_CON :
				CudaPropUtils::cuda_consistency_1b1c <<< _constraint_queue.size(), 1, _scope_state_size >>> ( _d_constraint_queue );
    			CudaSynchUtils::cuda_set_domains_from_bit_1b1v<<< _cp_model_ptr->num_variables (), 1, STANDARD_DOM >>> 
    			( 
    				_cp_model_ptr->get_dev_domain_index_ptr  (),
    				_cp_model_ptr->get_dev_domain_states_ptr () 
    			);
    			break;
    		default:
    			CudaPropUtils::cuda_consistency_sequential <<< 1, 1, _scope_state_size >>> ( _d_constraint_queue, _constraint_queue.size() );
    			break;
		}
	}
	else
	{
		CudaPropUtils::cuda_consistency_sequential <<< 1, 1, _scope_state_size >>> ( _d_constraint_queue, _constraint_queue.size() );
	}
	
#endif
}//dev_consistency



