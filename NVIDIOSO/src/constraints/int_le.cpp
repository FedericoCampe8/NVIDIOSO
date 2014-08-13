//
//  int_le.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 13/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "int_le.h"

IntLe::IntLe () :
FZNConstraint ( INT_LE ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  set_event( EventType::MIN_EVT );
  set_event( EventType::MAX_EVT );
}//IntLe

IntLe::IntLe ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
IntLe () {
  setup ( vars, args );
}//IntLe

IntLe::IntLe ( int x, int y ) :
FZNConstraint ( INT_LE ) {
  
  /*
   * Set x and y as arguments.
   * @note no FD variables here: scope size equal
   *       to 0 which is the default value.
   */
  _arguments.push_back( x );
  _arguments.push_back( y );
}//IntLe

IntLe::IntLe ( IntVariablePtr x, int y ) :
FZNConstraint ( INT_LE ) {
  
  // Consistency check on pointers
  if ( x == nullptr )
    throw NvdException ( (_dbg + "x variable is NULL").c_str() );
  
  // Assign the FD variable to _var_x
  _var_x = x;
  
  // Set the argument in the list of arguments
  _arguments.push_back( y );
  
  // One FD variable: scope size = 1;
  _scope_size = 1;
}//IntNe

IntLe::IntLe ( int x, IntVariablePtr y ) :
IntLe ( y, x ) {
}//IntLe

IntLe::IntLe ( IntVariablePtr x, IntVariablePtr y ) :
FZNConstraint ( INT_LE ) {
  
  // Consistency check on pointers
  if ( x == nullptr )
    throw NvdException ( (_dbg + "x variable is NULL").c_str() );
  if ( y == nullptr )
    throw NvdException ( (_dbg + "y variable is NULL").c_str() );
  
  // Common case: 2 FD variables
  _var_x = x;
  _var_y = y;
  _scope_size = 2;
}//IntLe

void
IntLe::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args ) {
  
  // Consistency checking in order to avoid more than one setup
  if ( (_var_x != nullptr) || (_arguments.size() > 0) ) return;
  
  if ( vars.size() == 0 ) {
    
    // Consistency check on args
    if ( args.size() != 2 )
      throw NvdException ( (_dbg + "wrong number of arguments (missing 2).").c_str() );
    
    // Set scope size (default: 0) ang arguments list
    _arguments.push_back( atoi ( args[ 0 ].c_str() ) );
    _arguments.push_back( atoi ( args[ 1 ].c_str() ) );
  }
  else if ( vars.size() == 1 ) {
    
    _var_x =
    std::dynamic_pointer_cast<IntVariable>( vars[ 0 ] );
    
    // Consistency check on pointers
    if ( _var_x == nullptr )
      throw NvdException ( (_dbg + "x variable is NULL").c_str() );
    
    // Consistency check on args
    if ( args.size() != 1 )
      throw NvdException ( (_dbg + "wrong number of arguments (missing 1).").c_str() );
    
    // Set scope size ang arguments list
    _scope_size = 1;
    _arguments.push_back( atoi ( args[ 0 ].c_str() ) );
  }
  else if ( vars.size() == 2 ) {
    
    _var_x =
    std::dynamic_pointer_cast<IntVariable>( vars[ 0 ] );
    
    _var_y =
    std::dynamic_pointer_cast<IntVariable>( vars[ 1 ] );
    
    // Consistency check on pointers
    if ( _var_x == nullptr )
      throw NvdException ( (_dbg + "x variable is NULL").c_str() );
    
    if ( _var_x == nullptr )
      throw NvdException ( (_dbg + "y variable is NULL").c_str() );
    
    // Set scope size ang arguments list
    _scope_size = 2;
  }
}//setup

const std::vector<VariablePtr>
IntLe::scope () const {
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  if ( _var_x != nullptr ) scope.push_back ( _var_x );
  if ( _var_y != nullptr ) scope.push_back ( _var_y );
  return scope;
}//scope

void
IntLe::consistency () {
  
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
    if ( !_var_x->is_singleton() ) {
      _var_x->in_max ( _arguments[ 0 ] );
    }
    return;
  }
  
  /*
   * 2 FD variables: if one is singleton,
   * propagate on the other.
   */
  if ( _scope_size == 2 ) {
    if ( (_var_x->is_singleton()) &&
        (!_var_y->is_singleton()) ) {
      _var_y->in_min ( _var_x->min () );
    }
    else if ( (_var_y->is_singleton()) &&
             (!_var_x->is_singleton()) ) {
      _var_x->in_max ( _var_y->min () );
    }
    return;
  }
}//consistency

//! It checks if x <= y
bool
IntLe::satisfied ()  {
  
  // No FD variables, just check the integers values
  if ( _scope_size == 0 ) {
    return _arguments[ 0 ] <= _arguments[ 1 ];
  }
  
  // 1 FD variable, if singleton check
  if ( (_scope_size == 1) &&
      _var_x->is_singleton() ) {
    return _arguments[ 0 ] <= _var_x->min ();
  }
  
  // 2 FD variables, if singleton check
  if ( (_scope_size == 2) &&
      (_var_x->is_singleton()) &&
      (_var_y->is_singleton()) ) {
    return _var_x->min () <= _var_y->min ();
  }
  
  /*
   * Check if a domain is empty.
   * If it is the case: failed propagation.
   */
  if ( _var_x->is_empty () ||
      _var_y->is_empty () )
    return false;
  
  /*
   * Other cases: there is not enough information
   * to state whether the constraint is satisfied or not.
   * Return true.
   */
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
IntLe::print_semantic () const {
  FZNConstraint::print_semantic();
  std::cout << "a <= b\n";
  std::cout << "int_le(var int: a, var int:b)\n";
}//print_semantic


