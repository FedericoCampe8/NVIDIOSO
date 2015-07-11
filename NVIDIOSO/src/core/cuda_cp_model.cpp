//
//  cuda_cp_model.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "cuda_cp_model.h"
#include "fzn_constraint.h"
#include "cuda_simple_constraint_store.h"
#include "cuda_constraint_factory.h"

#if CUDAON
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif 

using namespace std;

CudaCPModel::CudaCPModel () :
	_dbg 			   ( "CudaCPModel - "),
	_h_domain_states   ( nullptr ),
	_d_domain_states   ( nullptr ),
	_domain_state_size ( 0 )  {
}//CudaCPModel

CudaCPModel::~CudaCPModel () 
{
	// Free on host
  	free ( _h_domain_states );
  
#if CUDAON
	// Free on device
  	logger.cuda_handle_error ( cudaFree( _d_domain_states ) );
  	logger.cuda_handle_error ( cudaFree( _d_domain_index ) );
  	logger.cuda_handle_error ( cudaFree( d_constraint_description ) );
#endif

}//~CudaCPModel 
 
void
CudaCPModel::finalize () 
{	
	LogMsg << _dbg + "finalize: alloc vars and constraints on device" << std::endl;
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

bool
CudaCPModel::alloc_variables () 
{

#if CUDAON

	// Calculate total size to allocate on device
  	int var_id = 0;
  	for ( auto var : _variables ) 
  	{
    	_domain_state_size += ( (var->domain_iterator)->get_domain_status() ).first;
    	_cuda_var_lookup[ var->get_id() ] = var_id++;
  	}

  	// Allocate space on host and device
  	_h_domain_states = (uint*) malloc ( _domain_state_size );
  	if ( logger.cuda_handle_error ( cudaMalloc( (void**)&_d_domain_states, _domain_state_size ) ) ) 
  	{
    	return false;
  	}
  
  	// Set states on device
  	int idx = 0;
  	vector<int> map_vars_to_doms( _variables.size() );
  	for ( auto var : _variables ) 
  	{// Save the index where each variable domain starts at
    	map_vars_to_doms[ _cuda_var_lookup[ var->get_id() ] ] = idx;
    	idx += ( (var->domain_iterator)->get_domain_status() ).first / sizeof(int);
  	}
  
  	// Alloc lookup index for variables on device
  	if ( logger.cuda_handle_error ( cudaMalloc( (void**)&_d_domain_index, _variables.size() * sizeof ( int ) ) ) ) 
  	{
    	return false;
  	}
  
  	// Copy lookup index for variables on device
  	if ( logger.cuda_handle_error ( cudaMemcpy (_d_domain_index, &map_vars_to_doms[ 0 ],
    	                                        _variables.size() * sizeof ( int ), cudaMemcpyHostToDevice ) ) ) 
  	{
    	return false;
  	}
  	
	#endif
  
  return true;
}//alloc_variables

bool
CudaCPModel::alloc_constraints () 
{

#if CUDAON

  // Prepare info for cuda constraint factory
  int con_id = 0;
  vector<int> con_info;

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
    {
      // Note: id w.r.t. vars ids on device
      con_info.push_back ( _cuda_var_lookup [ var->get_id () ] );
    }
    
    // List of aux arguments
    for ( auto args : con->arguments () ) 
    {
      con_info.push_back ( args ); 
    }
    constraint_mapping_h_d [ con->get_unique_id () ] = con_id++;
  }//con

  // Copy above info on device
  if ( logger.cuda_handle_error ( cudaMalloc ((void**)&d_constraint_description,
                                               con_info.size() * sizeof (int) ) ) ) 
  {
    return false;
  }

  if ( logger.cuda_handle_error ( cudaMemcpy (d_constraint_description, &con_info[ 0 ],
                                               con_info.size() * sizeof (int),
                                               cudaMemcpyHostToDevice ) ) ) 
  {
    return false;
  }
  
  LogMsg << _dbg + "Instantiate constraints on device" << std::endl;
  // Create constraints on device
  CudaConstraintFactory::cuda_constrain_factory<<<1, 1>>> ( 
  d_constraint_description, 
  _constraints.size(),
  _d_domain_index,
  _d_domain_states );
  
  cudaDeviceSynchronize ();
  
#endif

  return true;
}//alloc_constraints

bool
CudaCPModel::upload_device_state () 
{

#if CUDAON

  int idx = 0;
  for ( auto var : _variables ) 
  { 
    	memcpy ( &_h_domain_states[idx], (uint*)( (var->domain_iterator)->get_domain_status() ).second,
            	 ( (var->domain_iterator)->get_domain_status() ).first );
            
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



