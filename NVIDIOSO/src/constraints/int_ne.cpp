//
//  int_ne.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 29/07/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "int_ne.h"

IntNe::IntNe ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::INT_NE );
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  set_event( EventType::SINGLETON_EVT );
}//IntNe

void
IntNe::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args ) {
  
  // Consistency checking in order to avoid more than one setup
  if ( (_var_x != nullptr) || (_arguments.size() > 0) ) return;
  
  if ( vars.size() == 0 ) 
  {  
    // Sanity check
    if ( args.size() != 2 )
      throw NvdException ( (_dbg + "wrong number of arguments (missing 2)").c_str() );
    
    // Set scope size (default: 0) ang arguments list
    _arguments.push_back( atoi ( args[ 0 ].c_str() ) );
    _arguments.push_back( atoi ( args[ 1 ].c_str() ) );
  }
  else if ( vars.size() == 1 ) 
  { 
    _var_x =
    std::dynamic_pointer_cast<IntVariable>( vars[ 0 ] );
    
    // Sanity check on pointers
    if ( _var_x == nullptr )
    {
    	throw NvdException ( (_dbg + "x variable is NULL").c_str() );
    }
    
    // Sanity check on args
    if ( args.size() != 1 )
    {
    	throw NvdException ( (_dbg + "wrong number of arguments (missing 1)").c_str() );
    }
    
    // Set scope size ang arguments list
    _scope_size = 1;
    _arguments.push_back( atoi ( args[ 0 ].c_str() ) );
  }
  else if ( vars.size() == 2 ) 
  {
    
    _var_x =
    std::dynamic_pointer_cast<IntVariable>( vars[ 0 ] );
    
    _var_y =
    std::dynamic_pointer_cast<IntVariable>( vars[ 1 ] );
    
    // Sanity check on pointers
    if ( _var_x == nullptr )
      throw NvdException ( (_dbg + "x variable is NULL").c_str() );
    
    if ( _var_x == nullptr )
      throw NvdException ( (_dbg + "y variable is NULL").c_str() );

    // Set scope size ang arguments list
    _scope_size = 2;
  }
}//setup

const std::vector<VariablePtr>
IntNe::scope () const 
{
  // Return constraint's scope
  std::vector<VariablePtr> scope;
  if ( _var_x != nullptr ) scope.push_back ( _var_x );
  if ( _var_y != nullptr ) scope.push_back ( _var_y );
  return scope;
}//scope

int 
IntNe::unsat_level () 
{
	//@todo implement level of unsat based on the values of the variables
	if ( satisfied () ) return 0;
	return 1;
}//unsat_level

void
IntNe::consistency () {

  /*
   * Propagate constraint iff there are two
   * FD variables and one is ground OR
   * there is one FD variable which is not ground.
   * @note (2 - _scope_size) can be used instead of
   *       get_arguments_size function since
   *       (2 - _scope_size) = get_arguments_size ().
   */
  if ( _scope_size == 0 ) return;
  
  // 1 FD variable: if not singleton, propagate.
  if ( _scope_size == 1 ) {
    if ( !_var_x->is_singleton() ) 
    {
      _var_x->subtract( _arguments[ 0 ] );
    }
    return;
  }
  
  /* 
   * 2 FD variables: if one is singleton,
   * propagate on the other.
   */
  if ( _scope_size == 2 ) 
  {
	if ( (_var_x->is_singleton()) && (!_var_y->is_singleton()) ) 
    {
    	_var_y->subtract( _var_x->min () );
    }
    else if ( (_var_y->is_singleton()) && (!_var_x->is_singleton()) ) 
    {
    	_var_x->subtract( _var_y->min () );
    }
    return;
  }
}//consistency

//! It checks if x != y
bool
IntNe::satisfied () 
{
  	// No FD variables, just check the integers values
  	if ( _scope_size == 0 ) 
  	{
    	return _arguments[ 0 ] != _arguments[ 1 ];
  	}
  	
  	// 1 FD variable, if singleton check
  	if ( (_scope_size == 1) &&
         _var_x->is_singleton() ) 
	{
    	return _arguments[ 0 ] != _var_x->min ();
  	}
  
  	// 2 FD variables, if singleton check
  	if ( (_scope_size == 2) 	  &&
       	 (_var_x->is_singleton()) &&
       	 (_var_y->is_singleton()) ) 
    {
    	return _var_x->min () != _var_y->min ();
  	}
  
	/*
   	 * Check if a domain is empty.
   	 * If it is the case: failed propagation.
   	 */
   	if ( _scope_size == 1 && _var_x->is_empty () )
	{
    	return false;
    }
    
  	if ( (_scope_size == 2) && (_var_x->is_empty () || _var_y->is_empty () ) )
	{	
    	return false;
    }

  	/*
   	 * Other cases: there is not enough information
   	 * to state whether the constraint is satisfied or not.
   	 * Return true.
   	 */
  	return true;
}//satisfied

//! Prints the semantic of this constraint
void
IntNe::print_semantic () const 
{
  BaseConstraint::print_semantic();
  std::cout << "a != b\n";
  std::cout << "int_ne(var int: a, var int:b)\n";
}//print_semantic



