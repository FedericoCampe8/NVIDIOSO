//
//  cuda_cp_model.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "cuda_cp_model.h"
#include "fzn_constraint.h"
#include "cuda_int_ne.h"
#include "cuda_int_lin_ne.h"
#include "cuda_simple_constraint_store.h"

#if CUDAON
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif 

using namespace std;

#if CUDAON
__device__ CudaConstraint** d_constraints_ptr;

__global__ void
cuda_print_constraints ( int n ) {
	for ( int i = 0; i < n; i++ )
  		d_constraints_ptr [ i ]->print();
}//cuda_print_constraints

__global__ void
cuda_constraint_factory ( int* constraint_description, size_t size,
                          int* domain_index, uint* domain_states ) {
  // Allocate memory for pointers to constraint classes
  d_constraints_ptr = (CudaConstraint**) malloc ( size * sizeof ( CudaConstraint* ) );
  
  // Create constriants on device
  int index = 0; 
  int c_id;
  int n_vars;
  int n_aux;
  int * vars;
  int * args;
  for ( int c = 0; c < size; c++ ) {
    c_id   = constraint_description  [ index + 1 ];
    n_vars = constraint_description  [ index + 2 ];
    n_aux  = constraint_description  [ index + 3 ];
    vars   = (int*) &constraint_description [ index + 4 ];
    args   = (int*) &constraint_description [ index + 4 + n_vars ];
    
    switch ( (FZNConstraintType) constraint_description[ index ] ) {
      case FZNConstraintType::INT_NE:
        d_constraints_ptr[ c ] = new CudaIntNe ( c_id, n_vars, n_aux, vars, args,
                                                 domain_index, domain_states );
        break;
        case FZNConstraintType::INT_LIN_NE:
        d_constraints_ptr[ c ] = new CudaIntLinNe ( c_id, n_vars, n_aux, vars, args,
                                                 	domain_index, domain_states );
        break;
      default:
        break;
    }
    
    index += 4 + n_vars + n_aux;
  }//c
}//cuda_constraint_factory
#endif

CudaCPModel::CudaCPModel () :
_h_domain_states   ( nullptr ),
_d_domain_states   ( nullptr ),
_domain_state_size ( 0 )  {
}//CudaCPModel

CudaCPModel::~CudaCPModel () {
  free ( _h_domain_states );
#if CUDAON
  logger->cuda_handle_error ( cudaFree( _d_domain_states ) );
  logger->cuda_handle_error ( cudaFree( _d_domain_index ) );
  logger->cuda_handle_error ( cudaFree( d_constraint_description ) );
#endif
}//~CudaCPModel 
 
void
CudaCPModel::finalize () {
	
	if ( (_variables.size   () == 0) ||
       (_constraints.size () == 0) ) 
	{
    	string err = "No variables or constraints to initialize on device.\n";
    	throw NvdException ( err.c_str(), __FILE__, __LINE__ );
  	}
  
  // Alloc variables and constraints on device
  if ( !alloc_variables   () ) 
  {
    string err = "Error in allocating variables on device.\n";
    throw NvdException ( err.c_str(), __FILE__, __LINE__ );
  }

  if ( !alloc_constraints () ) 
  {
    string err = "Error in allocating constraints on device.\n";
    throw NvdException ( err.c_str(), __FILE__, __LINE__ );
  }
  
  // Init constraint_store for cuda
  CudaSimpleConstraintStore* cuda_store_ptr =
    dynamic_cast<CudaSimpleConstraintStore* >( _store.get() );
  if ( cuda_store_ptr == nullptr ) {
    string err = "Error in casting the store (need CudaSimpleConstraintStore).\n";
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
CudaCPModel::alloc_variables () {
#if CUDAON
  
  // Calculate total size to allocate on device
  int var_id          = 0;
  for ( auto var : _variables ) {
    _domain_state_size += ( (var->domain_iterator)->get_domain_status() ).first;
    _cuda_var_lookup[ var->get_id() ] = var_id++;
  }

  // Allocate space on host and device
  _h_domain_states = (uint*) malloc ( _domain_state_size );
  if ( logger->cuda_handle_error ( cudaMalloc( (void**)&_d_domain_states, _domain_state_size ) ) ) {
    return false;
  }
  
  // Set states on device
  int idx = 0;
  vector<int> map_vars_to_doms( _variables.size() );
  for ( auto var : _variables ) {
    // Save the index where each variable domain starts at
    map_vars_to_doms[ _cuda_var_lookup[ var->get_id() ] ] = idx;
    memcpy ( &_h_domain_states[ idx ], (uint*)( (var->domain_iterator)->get_domain_status() ).second,
             ( (var->domain_iterator)->get_domain_status() ).first );
	
    idx += ( (var->domain_iterator)->get_domain_status() ).first / sizeof(int);
  }
  
  if ( logger->cuda_handle_error ( cudaMemcpy (_d_domain_states, &_h_domain_states[ 0 ],
                                               _domain_state_size, cudaMemcpyHostToDevice ) ) ) {
    return false;
  }
  
  if ( logger->cuda_handle_error ( cudaMalloc( (void**)&_d_domain_index, _variables.size() * sizeof ( int ) ) ) ) {
    return false;
  }
  if ( logger->cuda_handle_error ( cudaMemcpy (_d_domain_index, &map_vars_to_doms[ 0 ],
                                               _variables.size() * sizeof ( int ), cudaMemcpyHostToDevice ) ) ) {
    return false;
  }
#endif
  
  return true;
}//alloc_variables

bool
CudaCPModel::alloc_constraints () {

#if CUDAON

  // Prepare info for cuda constraint factory
  int con_id = 0;
  vector<int> con_info;

  for ( auto con : _constraints ) {
    // Constraint type
    con_info.push_back ( con->get_number_id () );
    
    // Constraint id
    con_info.push_back ( con->get_unique_id () );
    
    // Scope size
    con_info.push_back ( con->get_scope_size () );
    
    // Aux args size
    con_info.push_back ( con->get_arguments_size () );
    
    // List of variables involved in this constraint
    for ( auto var : con->scope () ) {
      // Note: id w.r.t. vars ids on device
      con_info.push_back ( _cuda_var_lookup [ var->get_id () ] );
    }
    
    // List of aux arguments
    for ( auto args : con->arguments () ) {
      con_info.push_back ( args ); 
    }
    constraint_mapping_h_d [ con->get_unique_id () ] = con_id++;
  }//con

  // Copy above info on device
  if ( logger->cuda_handle_error ( cudaMalloc ((void**)&d_constraint_description,
                                               con_info.size() * sizeof (int) ) ) ) {
    return false;
  }
  cudaDeviceSynchronize();
  if ( logger->cuda_handle_error ( cudaMemcpy (d_constraint_description, &con_info[ 0 ],
                                               con_info.size() * sizeof (int),
                                               cudaMemcpyHostToDevice ) ) ) {
    return false;
  }
  cudaDeviceSynchronize ();
  
  // Create constraints on device
  cuda_constraint_factory<<<1, 1>>> ( d_constraint_description, _constraints.size(),
                                      _d_domain_index, _d_domain_states );
                                      
#if CUDAVERBOSE
  cuda_print_constraints<<<1, 1>>> ( _constraints.size() );
#endif
  	cudaDeviceSynchronize ();
#endif

  return true;
}//alloc_constraints

bool
CudaCPModel::upload_device_state () {
#if CUDAON

  int idx = 0;
  for ( auto var : _variables ) {
    memcpy ( &_h_domain_states[idx], (uint*)( (var->domain_iterator)->get_domain_status() ).second,
            ( (var->domain_iterator)->get_domain_status() ).first );
            
    idx += ( (var->domain_iterator)->get_domain_status() ).first / sizeof(int); 
  }
  if ( logger->cuda_handle_error ( cudaMemcpy (_d_domain_states, &_h_domain_states[ 0 ],
                                               _domain_state_size, cudaMemcpyHostToDevice ) ) ) {
    string err = "Error updating device from host.\n";
    throw NvdException ( err.c_str(), __FILE__, __LINE__ );
  }
#endif

return true;
}//upload_device_state

bool
CudaCPModel::download_device_state () 
{
#if CUDAON

	if ( logger->cuda_handle_error ( cudaMemcpy (&_h_domain_states[ 0 ], _d_domain_states, 
                                            	_domain_state_size, cudaMemcpyDeviceToHost ) ) ) {
    	string err = "Error updating host from device.\n";
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



