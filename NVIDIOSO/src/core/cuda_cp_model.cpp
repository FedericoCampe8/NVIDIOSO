//
//  cuda_cp_model.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "cuda_cp_model.h"
#include "cuda_synchronization_utilities.h"
#include "cuda_simple_constraint_store.h"
#include "cuda_constraint_factory.h"

#if CUDAON
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif 

using namespace std;

CudaCPModel::CudaCPModel () :
    _dbg                 		  	 ( "CudaCPModel - "),
    _h_domain_states     			 ( nullptr ),
    _d_domain_states     			 ( nullptr ),
    _d_domain_states_aux 			 ( nullptr ),
    _d_base_constraint_description   ( nullptr ),
    _d_global_constraint_description ( nullptr ),
    _domain_state_size 				 ( 0 )  {
}//CudaCPModel

CudaCPModel::~CudaCPModel () 
{
    // Free on host
    free ( _h_domain_states );
  
#if CUDAON
	// Free on device
  	logger.cuda_handle_error ( cudaFree( _d_domain_states ) );
  	logger.cuda_handle_error ( cudaFree( _d_domain_index ) );
  	logger.cuda_handle_error ( cudaFree( _d_domain_states_aux ) );
  	logger.cuda_handle_error ( cudaFree( _d_base_constraint_description ) );
  	logger.cuda_handle_error ( cudaFree( _d_global_constraint_description ) );
  	logger.cuda_handle_error ( cudaFree( _d_additional_constraint_parameters ) );
  	logger.cuda_handle_error ( cudaFree( _d_additional_constraint_parameters_index ) );
  	logger.cuda_handle_error ( cudaFree( _d_additional_global_constraint_parameters ) );
  	logger.cuda_handle_error ( cudaFree( _d_additional_global_constraint_parameters_index ) );
#endif

}//~CudaCPModel 
 
uint * const 
CudaCPModel::get_dev_domain_states_ptr () const
{	
    return _d_domain_states;
}//get_dev_domain_states_ptr
  
int * const 
CudaCPModel::get_dev_domain_index_ptr () const
{
    return _d_domain_index;
}//get_dev_domain_index_ptr
 
uint * const 
CudaCPModel::get_dev_domain_states_aux_ptr () const 
{
 	return _d_domain_states_aux;
}//get_dev_domain_states_aux_ptr

void 
CudaCPModel::allocate_domain_states_aux ()
{

#if CUDAON

	if ( _d_domain_states_aux != nullptr || _domain_state_size == 0 )
	{
		return;
	}
	
	// Alloc lookup index for variables on device
  	if ( logger.cuda_handle_error ( cudaMalloc( (void**)&_d_domain_states_aux, _domain_state_size ) ) ) 
  	{
            string err = _dbg + "Couldn't allocate auxiliary array of states on device.\n";
            throw NvdException ( err.c_str(), __FILE__, __LINE__ );
  	}
  	
#endif

}//allocate_domain_states_aux 

void
CudaCPModel::device_state_to_aux ()
{

#if CUDAON
	if ( _d_domain_index == nullptr || _d_domain_states == nullptr || _d_domain_states_aux == nullptr )
	{
		string err = _dbg + "device_state_to_aux: NULL pointers\n";
        throw NvdException ( err.c_str(), __FILE__, __LINE__ );
	}

	CudaSynchUtils::cuda_copy_state_1b1v <<< num_variables (), 1 >>> 
	( _d_domain_index, _d_domain_states, _d_domain_states_aux );
	
#endif

}//device_state_to_aux

void
CudaCPModel::device_aux_to_state ()
{

#if CUDAON
	if ( _d_domain_index == nullptr || _d_domain_states == nullptr || _d_domain_states_aux == nullptr )
	{
		string err = _dbg + "device_aux_to_state: NULL pointers\n";
        throw NvdException ( err.c_str(), __FILE__, __LINE__ );
	}

	CudaSynchUtils::cuda_copy_state_1b1v <<< num_variables (), 1 >>> 
	( _d_domain_index, _d_domain_states_aux, _d_domain_states );
	
#endif

}//device_state_to_aux

void
CudaCPModel::finalize () 
{	
	LogMsg << _dbg + "finalize: allocating variables and constraints on device." << std::endl;
	if ( _variables.size () == 0 || _constraints.size () == 0 ) 
	{
            string err = _dbg + "No variables or constraints to initialize on device.\n";
            throw NvdException ( err.c_str(), __FILE__, __LINE__ );
  	}
  
  	// Alloc variables and constraints on device
  	if ( !alloc_variables   () ) 
  	{
            string err = _dbg +  "Error in allocating variables on device.\n";
            throw NvdException ( err.c_str(), __FILE__, __LINE__ );
  	}

  	if ( !alloc_constraints () ) 
  	{
            string err = _dbg +  "Error in allocating constraints on device.\n";
            throw NvdException ( err.c_str(), __FILE__, __LINE__ );
  	}
  
  	// Init constraint_store for cuda
  	CudaSimpleConstraintStore* cuda_store_ptr =
            dynamic_cast<CudaSimpleConstraintStore* >( _store.get() );
    	
  	if ( cuda_store_ptr == nullptr ) 
  	{
            string err = _dbg + "Error in casting the store (need CudaSimpleConstraintStore).\n";
            throw NvdException ( err.c_str(), __FILE__, __LINE__ );
  	}
  	try 
  	{
            cuda_store_ptr->finalize ( this );
  	} 
  	catch ( NvdException& e ) 
  	{
            throw e;
  	}
}//finalize

vector<int>
CudaCPModel::dev_var_mapping ( std::unordered_set<int> var_ids )
{
    vector<int> var_mapping;
    for ( auto var : var_ids )
    {
        var_mapping.push_back ( _map_vars_to_doms [ _cuda_var_lookup[ var ] ] );
    }
    return var_mapping;
}//dev_var_mapping

bool
CudaCPModel::alloc_variables () 
{

#if CUDAON

	/*
	 * Calculate total size to allocate on device,
	 * and store the index where each variable's domain starts at.
	 */
	_map_vars_to_doms.resize ( _variables.size() );

	int idx    = 0;
  	int var_id = 0;
  	for ( auto var : _variables ) 
  	{
            _domain_state_size += ( (var->domain_iterator)->get_domain_status() ).first;
            _cuda_var_lookup[ var->get_id() ] = var_id++;
            _map_vars_to_doms[ _cuda_var_lookup[ var->get_id() ] ] = idx;
            idx += ( (var->domain_iterator)->get_domain_status() ).first / sizeof(int);
  	}

  	// Allocate memory for variables on host
  	_h_domain_states = (uint*) malloc ( _domain_state_size );
  
  	// ================   Upload variable information on device   ================
  	
  	// Allocate memory for variables on device
  	LogMsg << _dbg + "Allocate memory for variables on device" << std::endl;
  	if ( logger.cuda_handle_error ( cudaMalloc( (void**)&_d_domain_states, _domain_state_size ) ) ) 
  	{
            return false;
  	}

  	// Alloc lookup index for variables on device
  	if ( logger.cuda_handle_error ( cudaMalloc( (void**)&_d_domain_index, _variables.size() * sizeof ( int ) ) ) ) 
  	{
            return false;
  	}

  	// Copy lookup index for variables on device
  	LogMsg << _dbg + "Copy variables information on device" << std::endl;
  	if ( logger.cuda_handle_error ( cudaMemcpy (_d_domain_index, &_map_vars_to_doms[ 0 ],
                                                _variables.size() * sizeof ( int ), cudaMemcpyHostToDevice ) ) )
	{ 
            return false;
  	}

	#endif
	
  return true;
}//alloc_variables

bool
CudaCPModel::alloc_base_constraints () 
{
	
#if CUDAON

	//Sanity check
	if ( _constraints.size () == 0 ) return true;
	
	// Prepare info for cuda constraint factory
  	int con_id = 0;
  	int con_additional_parameters_size = 0;
  	std::vector<int> con_info;
  	std::vector<int> con_additional_parameters;
  	std::vector<int> con_additional_parameters_index;
  	
  	for ( auto con : _constraints ) 
  	{
    	// Constraint type
    	con_info.push_back ( con->get_number_id () );
    
    	// Constraint id
    	con_info.push_back ( con->get_unique_id () );
    
    	// Scope size
    	con_info.push_back ( con->get_scope_size () );
    
    	// Aux args size
    	con_info.push_back ( con->get_arguments_size () );
    
    	// List of variables involved in this constraint
    	for ( auto var : con->scope () ) 
    	{// Note: id w.r.t. vars ids on device
        	con_info.push_back ( _cuda_var_lookup [ var->get_id () ] );
    	}
    
    	// List of non variable arguments
    	for ( auto args : con->arguments () ) 
    	{
        	con_info.push_back ( args ); 
    	}
    	
    	/*
    	 * Set shared arguments between constraints as additional constraint parameters.
    	 * @note base constrains have only 1 shared arguments array.
    	 */
    	if ( con->get_number_shared_arguments () > 0 )
    	{
    		con_additional_parameters_index.push_back ( con_additional_parameters_size );
    		con_additional_parameters_size += (con->get_shared_arguments()).size(); 
    		for ( auto& x : con->get_shared_arguments() )
    		{
    			con_additional_parameters.push_back ( x );
    		}
    	}
    	else
    	{
    		con_additional_parameters_index.push_back ( -1 );
    	}
    	
    	constraint_mapping_h_d [ con->get_unique_id () ] = con_id++;
  	}//con
  	
  	// ================   Copy constraint information on device   ================
  	// Allocate memory for constraint description
  	LogMsg << _dbg + "Allocate memory for constraints on device" << std::endl;
  	if ( logger.cuda_handle_error ( cudaMalloc ((void**)&_d_base_constraint_description,
                                                con_info.size() * sizeof (int) ) ) ) 
  	{
    	return false;
  	}

  	// Allocate memory for additional parameters - Indexes
  	if ( logger.cuda_handle_error ( cudaMalloc ((void**)&_d_additional_constraint_parameters_index,
                                                con_additional_parameters_index.size() * sizeof (int) ) ) ) 
  	{
    	return false;
  	}
  
  	// Allocate memory for additional parameters - Values
  	if ( logger.cuda_handle_error ( cudaMalloc ((void**)&_d_additional_constraint_parameters,
                                                con_additional_parameters.size() * sizeof (int) ) ) ) 
  	{
    	return false;
  	}
  
  	// Copy for constraint description on device
  	LogMsg << _dbg + "Copy constraint information on device" << std::endl;
  	if ( logger.cuda_handle_error ( cudaMemcpy ( _d_base_constraint_description, 
  												 &con_info[ 0 ],
        	                                     con_info.size() * sizeof (int),
            	                                 cudaMemcpyHostToDevice ) ) ) 
  	{
    	  return false;
  	}
  
  	// Copy additional parameters - Indexes
  	if ( logger.cuda_handle_error ( cudaMemcpy ( _d_additional_constraint_parameters_index, 
  												 &con_additional_parameters_index[ 0 ],
        	                                     con_additional_parameters_index.size() * sizeof (int),
            	                                 cudaMemcpyHostToDevice ) ) ) 
  	{
    	  return false;
 	}
  
  	// Copy additional parameters - Values
  	if ( logger.cuda_handle_error ( cudaMemcpy ( _d_additional_constraint_parameters, 
  												 &con_additional_parameters[ 0 ],
        	                                     con_additional_parameters.size() * sizeof (int),
            	                                 cudaMemcpyHostToDevice ) ) ) 
  	{
      	return false;
  	}

#endif

	return true;
	
}//alloc_base_constraints

bool
CudaCPModel::alloc_global_constraints () 
{
	
#if CUDAON

	//Sanity check
	if ( _glb_constraints.size () == 0 ) return true;
	
	// Prepare info for cuda constraint factory
  	int con_id = 0;
  	int con_additional_parameters_size = 0;
  	std::vector<int> con_info;
  	std::vector<int> con_additional_parameters;
  	std::vector<int> con_additional_parameters_index;
  	
  	for ( auto con : _glb_constraints ) 
  	{
    	// Constraint type
    	con_info.push_back ( (int) ( con->get_global_constraint_type () ) );
    
    	// Constraint id
    	con_info.push_back ( con->get_unique_id () );
    
    	// Scope size
    	con_info.push_back ( con->get_scope_size () );
    
    	// Aux args size
    	con_info.push_back ( con->get_arguments_size () );
    
    	// List of variables involved in this constraint
    	for ( auto var : con->scope () ) 
    	{// Note: id w.r.t. vars ids on device
        	con_info.push_back ( _cuda_var_lookup [ var->get_id () ] );
    	}
    
    	// List of non variable arguments
    	for ( auto args : con->arguments () ) 
    	{
        	con_info.push_back ( args ); 
    	}
    	
    	/*
    	 * Set shared arguments between constraints as additional constraint parameters.
    	 * @note base constrains have only 1 shared arguments array.
    	 */
    	if ( con->get_number_shared_arguments () > 0 )
    	{
    		con_additional_parameters_index.push_back ( con_additional_parameters_size );
    		con_additional_parameters_size += (con->get_shared_arguments()).size(); 
    		for ( auto& x : con->get_shared_arguments() )
    		{
    			con_additional_parameters.push_back ( x );
    		}
    	}
    	else
    	{
    		con_additional_parameters_index.push_back ( -1 );
    	}
    	
    	constraint_mapping_h_d [ con->get_unique_id () ] = con_id++;
  	}//con
  	
  	// ================   Copy constraint information on device   ================
  	// Allocate memory for constraint description
  	LogMsg << _dbg + "Allocate memory for constraints on device" << std::endl;
  	if ( logger.cuda_handle_error ( cudaMalloc ((void**)&_d_global_constraint_description,
                                                con_info.size() * sizeof (int) ) ) ) 
  	{
    	return false;
  	}

  	// Allocate memory for additional parameters - Indexes
  	if ( logger.cuda_handle_error ( cudaMalloc ((void**)&_d_additional_global_constraint_parameters_index,
                                                con_additional_parameters_index.size() * sizeof (int) ) ) ) 
  	{
    	return false;
  	}
  
  	// Allocate memory for additional parameters - Values
  	if ( logger.cuda_handle_error ( cudaMalloc ((void**)&_d_additional_global_constraint_parameters,
                                                con_additional_parameters.size() * sizeof (int) ) ) ) 
  	{
    	return false;
  	}
  
  	// Copy for constraint description on device
  	LogMsg << _dbg + "Copy constraint information on device" << std::endl;
  	if ( logger.cuda_handle_error ( cudaMemcpy ( _d_global_constraint_description, 
  												 &con_info[ 0 ],
        	                                     con_info.size() * sizeof (int),
            	                                 cudaMemcpyHostToDevice ) ) ) 
  	{
    	  return false;
  	}
  
  	// Copy additional parameters - Indexes
  	if ( logger.cuda_handle_error ( cudaMemcpy ( _d_additional_global_constraint_parameters_index, 
  												 &con_additional_parameters_index[ 0 ],
        	                                     con_additional_parameters_index.size() * sizeof (int),
            	                                 cudaMemcpyHostToDevice ) ) ) 
  	{
    	  return false;
 	}
  
  	// Copy additional parameters - Values
  	if ( logger.cuda_handle_error ( cudaMemcpy ( _d_additional_global_constraint_parameters, 
  												 &con_additional_parameters[ 0 ],
        	                                     con_additional_parameters.size() * sizeof (int),
            	                                 cudaMemcpyHostToDevice ) ) ) 
  	{
      	return false;
  	}

#endif

	return true;
	
}//alloc_global_constraints


bool
CudaCPModel::alloc_constraints () 
{

#if CUDAON
	
	bool valid_allocation = true;
	valid_allocation &= alloc_base_constraints   ();
	valid_allocation &= alloc_global_constraints ();

	// =========== BASE CONSTRAINTS ON DEVICE ===========
	if ( _d_base_constraint_description != nullptr )
	{
		LogMsg << _dbg + "Instantiate base constraints on device" << std::endl;
  		CudaConstraintFactory::cuda_constrain_factory<<<1, 1>>> 
  		( 
			_d_base_constraint_description,
    		_constraints.size(),
    		_d_domain_index,
    		_d_domain_states,
    		_d_additional_constraint_parameters_index,
    		_d_additional_constraint_parameters
 		);
 	}
 	
 	// =========== GLOBAL CONSTRAINTS ON DEVICE ===========
 	if ( _d_global_constraint_description != nullptr )
	{
		LogMsg << _dbg + "Instantiate global constraints on device" << std::endl;
		CudaConstraintFactory::cuda_global_constrain_factory<<<1, 1>>> 
  		( 
			_d_global_constraint_description,
    		_glb_constraints.size(),
    		_d_domain_index,
    		_d_domain_states,
    		_d_additional_global_constraint_parameters_index,
    		_d_additional_global_constraint_parameters
 		);
 	}
  
	// Synchronization point
  	cudaDeviceSynchronize ();
  
#endif

  return true;
}//alloc_constraints

void
CudaCPModel::reset_device_state ()
{
}//reset_device_state

bool
CudaCPModel::upload_device_state () 
{

#if CUDAON

	int idx = 0;
  	for ( auto var : _variables ) 
  	{ 
		// Copy on _h_domain_states domains of all variables (i.e., current search status)
      	memcpy ( &_h_domain_states[idx], 
      		   	 (uint*)( (var->domain_iterator)->get_domain_status() ).second,
               	 (var->domain_iterator->get_domain_status()).first );
      
      	idx += ( (var->domain_iterator)->get_domain_status() ).first / sizeof(int); 
  	}
  	if ( logger.cuda_handle_error ( cudaMemcpy (_d_domain_states, &_h_domain_states[ 0 ],
                                              	_domain_state_size, cudaMemcpyHostToDevice ) ) ) 
  	{
      	string err = _dbg + "Error updating device from host.\n";
      	throw NvdException ( err.c_str(), __FILE__, __LINE__ );
  	}
  
#endif

return true;
}//upload_device_state

bool
CudaCPModel::download_device_state () 
{

#if CUDAON

    if ( logger.cuda_handle_error ( cudaMemcpy (&_h_domain_states[ 0 ], _d_domain_states, 
                                            	_domain_state_size, cudaMemcpyDeviceToHost ) ) ) 
    {
    	string err = _dbg + "Error updating host from device.\n";
    	throw NvdException ( err.c_str(), __FILE__, __LINE__ );
    }

    int idx = 0;
    for ( auto var : _variables ) 
    {  		
        (var->domain_iterator)->set_domain_status( (void *) &_h_domain_states[ idx ] );
        idx += ( (var->domain_iterator)->get_domain_status() ).first / sizeof(int);
        if ( var->is_empty() ) return false;
    }
    
#endif

return true;
}//download_device_state



