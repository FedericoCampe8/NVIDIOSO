//
//  cuda_simple_constraint_store.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 04/12/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//


#include "cuda_simple_constraint_store.h"
#include "cuda_propagation_utilities.h"

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

  	int i = 0;
  	for ( auto c: _constraint_queue ) 
  	{
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

  	// Propagate constraints in parallel
  	cuda_consistency<<< _constraint_queue.size(), 1 >>> ( _d_constraint_queue );
	cudaDeviceSynchronize (); 

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


