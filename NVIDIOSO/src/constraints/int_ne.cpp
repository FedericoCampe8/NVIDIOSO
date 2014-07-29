//
//  int_ne.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 29/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "int_ne.h"
#include "cuda_variable.h"

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
  
  // Common case: 2 FD variables
  _var_x = x;
  _var_y = y;
  _scope_size = 2;
}//IntNe

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
  std::cout << "a != b\n";
  std::cout << "int_ne(var int: a, var int:b)\n";
}//print_semantic



