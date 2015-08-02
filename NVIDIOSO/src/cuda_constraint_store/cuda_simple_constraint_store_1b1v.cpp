//
//  cuda_simple_constraint_store_1b1v.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/21/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//


#include "cuda_simple_constraint_store_1b1v.h"
#include "cuda_propagation_utilities.h"
#include "cuda_synchronization_utilities.h"

using namespace std;

CudaSimpleConstraintStore1b1v::CudaSimpleConstraintStore1b1v () :
    CudaSimpleConstraintStore (),
    _h_constraint_queue_size  ( 0 ),
    _h_con_idx_queue_size     ( 0 ),
    _d_states_aux             ( nullptr ) {
    _dbg = "CudaSimpleConstraintStore1b1v - ";
}//CudaSimpleConstraintStore1b1v

CudaSimpleConstraintStore1b1v::~CudaSimpleConstraintStore1b1v () {
    
#if CUDAON
    
    cudaFree ( _d_constraint_queue_idx );
    
#endif
    
}//~CudaSimpleConstraintStore1b1v

void
CudaSimpleConstraintStore1b1v::finalize ( CudaCPModel* ptr )
{
    CudaSimpleConstraintStore::finalize ( ptr );
    if ( _cp_model_ptr == nullptr )
    {
        std::string err = _dbg + "finalize: CudaCPModel NULL.\n";
        LogMsg << err;
        throw NvdException ( err.c_str(), __FILE__, __LINE__ );
    }

    _h_constraint_queue_size = _cp_model_ptr->num_constraints ();

#if CUDAON
    if ( logger.cuda_handle_error (
             cudaMalloc ( (void**)&_d_constraint_queue_idx,
                          (_cp_model_ptr->num_variables () + 1) * sizeof ( int )) ) )
    {
        string err = _dbg + "finalize: Bad memory allocation on device.\n";
        throw NvdException ( err.c_str(), __FILE__, __LINE__ );
    }
#endif
	
	// Allocate auxiliary states on device to synchronize threads during propagation
	_cp_model_ptr->allocate_domain_states_aux ();
	
	_d_states_aux = _cp_model_ptr->get_dev_domain_states_aux_ptr ();
	
    reset ();
}//finalize

void
CudaSimpleConstraintStore1b1v::add_changed ( size_t c_id, EventType event )
{
    CudaSimpleConstraintStore::add_changed ( c_id, event );

    int var_idx = 0, scope_size = 0;
    auto c_scope = _lookup_table[ c_id ]->scope ();
    for ( auto& var : c_scope )
    {
        // Do not propagate on singleton variables
        if ( var->is_singleton () )
        {
            var_idx++;
            continue;
        }
        
        _var_to_constraint [ var->get_id () ].push_back ( _cp_model_ptr->constraint_mapping_h_d[ c_id ] );
        _var_to_constraint [ var->get_id () ].push_back ( var_idx++ );
        _h_con_idx_queue_size += 2;
        scope_size = ( _lookup_table.find( c_id )->second->get_scope_size() * STANDARD_DOM * sizeof (int) );
        if ( _scope_state_size < scope_size )
        {
            _scope_state_size = scope_size;
        }
    }
}//add_changed

bool 
CudaSimpleConstraintStore1b1v::move_states_to_device ()
{	
	if ( _d_states_aux == nullptr )
	{
		string err = _dbg + "move_states_to_device: auxiliary array for states NULL.\n";
        throw NvdException ( err.c_str(), __FILE__, __LINE__ );
	}

	bool upload = CudaSimpleConstraintStore::move_states_to_device ();
	 
	//! Copy state to auxiliary array
	_cp_model_ptr->device_state_to_aux ();

	return upload;
}//move_states_to_device

bool 
CudaSimpleConstraintStore1b1v::move_states_from_device () 
{
	if ( _d_states_aux == nullptr )
	{
		string err = _dbg + "move_states_from_device: auxiliary array for states NULL.\n";
        throw NvdException ( err.c_str(), __FILE__, __LINE__ );
	}
	
	//! Copy state from auxiliary array
	_cp_model_ptr->device_aux_to_state ();
	
	bool download = CudaSimpleConstraintStore::move_states_from_device ();
	return download;
}//move_states_from_device

void
CudaSimpleConstraintStore1b1v::move_queue_to_device ()
{
    
#if CUDAON

    // Prepare array of constraints and indeces to copy on device
    _h_constraint_queue.clear ();
    _h_constraint_queue_idx.clear ();
	
    int idx = 0;
    for ( auto& var : _var_to_constraint )
    {
        _h_constraint_queue_idx.push_back ( idx );
        for ( auto& c_id : var.second )
        {
        	_h_constraint_queue.push_back ( c_id );
            ++idx;
        }
    }
    _h_constraint_queue_idx.push_back ( idx );
    
    // Check is the queue on device if large enough to contain all constraints
    if ( _h_con_idx_queue_size >= _h_constraint_queue_size && _scope_state_size <= _shared_limit )
    {// If not, allocate more memory 

        cudaFree ( _d_constraint_queue );
        if ( logger.cuda_handle_error (
        cudaMalloc ( (void**)&_d_constraint_queue, _h_con_idx_queue_size * sizeof ( size_t )) ) )
        {
            string err = _dbg + "move_queue_to_device: Bad memory allocation on device.\n";
            throw NvdException ( err.c_str(), __FILE__, __LINE__ );
        }
    }
	
    // Copy constraints idx per var on device
    if ( _scope_state_size <= _shared_limit )
    {
    	cudaMemcpy( _d_constraint_queue, &_h_constraint_queue[0],
        	        _h_con_idx_queue_size * sizeof( size_t ), cudaMemcpyHostToDevice );
    
    	cudaMemcpy( _d_constraint_queue_idx, &_h_constraint_queue_idx[0],
         	        _h_constraint_queue_idx.size() * sizeof( int ), cudaMemcpyHostToDevice );
        
    	dev_grid_size.x = _h_constraint_queue_idx.size() - 1;
    }
    
#endif

}//move_queue_to_device

void
CudaSimpleConstraintStore1b1v::reset ()
{
	_scope_state_size     = 0;
    _h_con_idx_queue_size = 0;
    _var_to_constraint.clear();
}//reset

void
CudaSimpleConstraintStore1b1v::sequential_propagation ()
{

    #if CUDAON

    static bool advice = true;
    if ( advice )
    {
        std::cout << _dbg + "Shared memory limit on device reached:\n";
        std::cout << "The current version of iNVIDIOSO uses sequential propagation\n";
        std::cout << "on device when the limit of " << _shared_limit << " bytes\n";
        std::cout << "for shared memory is reached.\n";
        std::cout << "Press any key to continue (this message won't be dysplayed again)\n";
        getchar();
        advice = false;
    }
    CudaPropUtils::cuda_consistency_sequential <<< 1, 1, _scope_state_size >>> 
    ( _d_constraint_queue, _constraint_queue.size() );

    #endif

}//sequential_propagation

void
CudaSimpleConstraintStore1b1v::dev_consistency ()
{

#if CUDAON

    if ( _scope_state_size > _shared_limit )
    {
        sequential_propagation ();
    }
    else
    {
        // Propagate constraints in parallel
        CudaPropUtils::cuda_consistency_1b1v <<< dev_grid_size, dev_block_size, _scope_state_size >>> 
        ( _d_constraint_queue, _d_constraint_queue_idx, STANDARD_DOM, _d_states_aux );
        
        // Reset mappings for next propagation
        reset ();

        dev_grid_size.x  = _cp_model_ptr->num_variables ();
        CudaSynchUtils::cuda_set_domains_from_bit_1b1v <<< dev_grid_size, dev_block_size, STANDARD_DOM * sizeof ( uint ) >>>
        ( _cp_model_ptr->get_dev_domain_index_ptr (), _d_states_aux );
    }
    
#endif
}//dev_consistency



