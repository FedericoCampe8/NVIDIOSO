//
//  int_ne.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 29/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "int_ne.h"
#include "cuda_variable.h"

IntNe::IntNe () :
FZNConstraint ( INT_NE ) {
}//IntNe

IntNe::IntNe ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
IntNe () {
  setup ( vars, args );
}//IntNe

IntNe::IntNe ( int x, int y ) :
FZNConstraint ( INT_NE ) {
  
  /*
   * Set x and y as arguments.
   * @note no FD variables here: scope size equal
   *       to 0 which is the default value.
   */
  _arguments.push_back( x );
  _arguments.push_back( y );
}//IntNe

IntNe::IntNe ( IntVariablePtr x, int y ) :
FZNConstraint ( INT_NE ) {
  
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

IntNe::IntNe ( int x, IntVariablePtr y ) :
IntNe ( y, x ) {
}//IntNe

IntNe::IntNe ( IntVariablePtr x, IntVariablePtr y ) :
FZNConstraint ( INT_NE ) {
  
  // Consistency check on pointers
  if ( x == nullptr )
    throw NvdException ( (_dbg + "x variable is NULL").c_str() );
  if ( y == nullptr )
    throw NvdException ( (_dbg + "y variable is NULL").c_str() );

  // Common case: 2 FD variables
  _var_x = x;
  _var_y = y;
  _scope_size = 2;
}//IntNe

void
IntNe::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args ) {
  
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
    std::static_pointer_cast<IntVariable>( vars[ 0 ] );
    
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
    std::static_pointer_cast<IntVariable>( vars[ 0 ] );
    
    _var_y =
    std::static_pointer_cast<IntVariable>( vars[ 1 ] );
    
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
IntNe::scope () const {
  
  // Return the scope of this constraint
  std::vector<VariablePtr> scope;
  scope.push_back ( _var_x );
  scope.push_back ( _var_y );
  return scope;
}//scope

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
  if ( get_arguments_size() == 2 ) return;
  
  // 1 FD variable: if not singleton, propagate.
  if ( get_arguments_size() == 1 ) {
    if ( !_var_x->is_singleton() ) {
      _var_x->subtract( _arguments[ 0 ] );
    }
    return;
  }
  
  /* 
   * 2 FD variables: if one is singleton,
   * propagate on the other.
   */
  if ( get_arguments_size() == 2 ) {
    if ( (_var_x->is_singleton()) &&
         (!_var_y->is_singleton()) ) {
      _var_y->subtract( _var_x->min() );
    }
    else if ( (_var_y->is_singleton()) &&
              (!_var_x->is_singleton()) ) {
      _var_x->subtract( _var_y->min() );
    }
    return;
  }
}//consistency

//! It checks if x != y
bool
IntNe::satisfied ()  {
  
  // No FD variables, just check the integers values
  if ( _arguments.size() == 2 ) {
    return _arguments[ 0 ] != _arguments[ 1 ];
  }
  
  // 1 FD variable, if singleton check
  if ( (_arguments.size() == 1) &&
       _var_x->is_singleton() ) {
    return _arguments[ 0 ] != _var_x->min();
  }
  
  // 2 FD variables, if singleton check
  if ( (_arguments.size() == 0) &&
       (_var_x->is_singleton()) &&
       (_var_y->is_singleton()) ) {
    return _var_x->min() != _var_y->min();
  }
  
  /*
   * Other cases: there is not enough information
   * to state whether the constraint is satisfied.
   * Return true.
   */
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
IntNe::print_semantic () const {
  FZNConstraint::print_semantic();
  std::cout << "a != b\n";
  std::cout << "int_ne(var int: a, var int:b)\n";
}//print_semantic



