//
//  cuda_model_generator.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 09/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "cuda_model_generator.h"
#include "cuda_variable.h"
#include "token_var.h"

using namespace std;

CudaGenerator::CudaGenerator () :
_dbg( "CudaGenerator - " ){
  logger->message ( _dbg + "Instantiate a generator for CUDA." );
}//CudaGenerator

CudaGenerator::~CudaGenerator () {
}//~CudaGenerator

VariablePtr
CudaGenerator::get_variable ( TokenPtr tkn_ptr ) {
  
  // Assert current token not nullptr
  assert( tkn_ptr != nullptr );
  
  // Check consistency of current token
  if ( (tkn_ptr->get_type() != TokenType::FD_VARIABLE) &&
       (tkn_ptr->get_type() != TokenType::FD_VAR_ARRAY) ) {
    logger->error( _dbg + "Error while instantiating a FD Variables",
                  __FILE__, __LINE__);
    throw new std::string ( "Error while instantiating a FD Variables" );
  }

  // Token (pointer) to return
  VariablePtr var_ptr = make_shared<CudaVariable> ( tkn_ptr->get_id() );
  
  // Set string id
  var_ptr->set_str_id( (std::static_pointer_cast<TokenVar>( tkn_ptr ))->get_var_id () );
  
  // Set type
  if ( std::static_pointer_cast<TokenVar>( tkn_ptr )->is_objective_var () ) {
    var_ptr->set_type( VariableType::OBJ_VARIABLE );
  }
  else if ( std::static_pointer_cast<TokenVar>( tkn_ptr )->is_support_var () ) {
    var_ptr->set_type( VariableType::SUP_VARIABLE );
  }
  else {
    var_ptr->set_type( VariableType::FD_VARIABLE );
  }
  
  /*
   * Set domain.
   * @note: only int (range) domains on CUDA.
   * @note: tkn_ptr is not given as input to the function "set_domain".
   *        This is done in order to decouple tokens from domains.
   */
  var_ptr->set_domain ( DomainType::INTEGER );
  switch ( (std::static_pointer_cast<TokenVar>( tkn_ptr ))->get_var_dom_type() ) {
    case VarDomainType::BOOLEAN:
      (std::static_pointer_cast<CudaVariable>( var_ptr ))->set_domain( 0, 1 );
      break;
    case VarDomainType::INTEGER:
      (std::static_pointer_cast<CudaVariable>( var_ptr ))->set_domain ();
      break;
    case VarDomainType::FLOAT:
      (std::static_pointer_cast<CudaVariable>( var_ptr ))->set_domain ();
      break;
    case VarDomainType::RANGE:
      (std::static_pointer_cast<CudaVariable>( var_ptr ))->
      set_domain (
                  (std::static_pointer_cast<TokenVar>( tkn_ptr ))->
                  get_lw_bound_domain(),
                  (std::static_pointer_cast<TokenVar>( tkn_ptr ))->
                  get_up_bound_domain()
                  );
      break;
    case VarDomainType::SET:
      (std::static_pointer_cast<CudaVariable>( var_ptr ))->
      set_domain (
                  (std::static_pointer_cast<TokenVar>( tkn_ptr ))->
                  get_subset_domain()
                  );
      break;
    case VarDomainType::SET_INT:
      (std::static_pointer_cast<CudaVariable>( var_ptr ))->
      set_domain (
                  (std::static_pointer_cast<TokenVar>( tkn_ptr ))->
                  get_subset_domain()
                  );
      break;
    case VarDomainType::SET_RANGE:
      (std::static_pointer_cast<CudaVariable>( var_ptr ))->
      set_domain (
                  (std::static_pointer_cast<TokenVar>( tkn_ptr ))->
                  get_subset_domain()
                  );
      break;
    default:
      /*
       * Return a nullptr to check in order to further
       * specilize this method if other types will be added.
       */
      return nullptr;
      break;
  }
  
  return var_ptr;
}//get_variable

ConstraintPtr
CudaGenerator::get_constraint ( TokenPtr tkn_ptr ) {
  
  // Error handling
  assert( tkn_ptr != nullptr );
  
  if ( tkn_ptr->get_type() != TokenType::FD_CONSTRAINT ) {
    logger->error( _dbg + "Error while instantiating a Constraint",
                  __FILE__, __LINE__);
    return nullptr;
  }
  
  return nullptr;
}//get_constraint

SearchEnginePtr
CudaGenerator::get_search_engine ( TokenPtr tkn_ptr ) {
  
  // Error handling
  assert( tkn_ptr != nullptr );
  
  if ( tkn_ptr->get_type() != TokenType::FD_SOLVE ) {
    logger->error( _dbg + "Error while instantiating a Search Engine",
                   __FILE__, __LINE__);
    return nullptr;
  }
  
  return nullptr;
}//get_search_engine




