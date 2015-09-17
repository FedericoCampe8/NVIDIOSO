//
//  bool_2_int.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "bool_2_int.h"

Bool2Int::Bool2Int ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::BOOL2INT );
	/*
	* Set the event that trigger this constraint.
	* @note if no event is set, this constraint will never be re-evaluated.
	*/
    set_event( EventType::SINGLETON_EVT );
	set_event( EventType::MIN_EVT );
	set_event( EventType::MAX_EVT );
	set_event( EventType::BOUNDS_EVT );
}//Bool2Int

void
Bool2Int::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args ) {
	
	// Consistency checking in order to avoid more than one setup
    if ( (_var_x != nullptr) || (_arguments.size() > 0) ) return;

    if ( vars.size() == 0 ) {

    // Consistency check on args
    if ( args.size() != 2 )
      throw NvdException ( (_dbg + "wrong number of arguments (missing 2).").c_str() );

    // Set scope size and arguments list
    _arguments.push_back( atoi ( args[ 0 ].c_str() ) );
    _arguments.push_back( atoi ( args[ 1 ].c_str() ) );

    _scope_size = 0;
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

        // Set scope size and arguments list
        _arguments.push_back( atoi ( args[ 0 ].c_str() ) );

        _scope_size = 1;
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

        // Set scope size and arguments list
        _scope_size = 2;
	}
}//setup

const std::vector<VariablePtr>
Bool2Int::scope () const {
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  if ( _var_x != nullptr ) scope.push_back ( _var_x );
  if ( _var_y != nullptr ) scope.push_back ( _var_y );
  return scope;
}//scope

void
Bool2Int::consistency () {
	
	// No variables: no propagations
    if ( _scope_size == 0 ) return;

    // One variable not singleton: propagate on it
    if ( _scope_size == 1 &&
         !_var_x->is_singleton() ) {
            _var_x->shrink ( _arguments[ 0 ], _arguments[ 0 ] );
         }
         return;

    /*
     * Two variables: propagate on variables that
     * aren't singletons
     */
     if ( _scope_size == 2 ) {
        if ( !_var_x->is_singleton() &&
             !_var_y->is_singleton() ) {
            _var_x->in_min ( _var_y->min() );
            _var_x->in_max ( _var_y->max() );
            _var_y->in_min ( _var_x->min() );
            _var_y->in_max ( _var_x->max() );
        } else if ( !_var_x->is_singleton() ) {
            _var_x->shrink( _var_y->min(), _var_y->min() );
        } else if ( !_var_y->is_singleton() ) {
            _var_y->shrink( _var_x->min(), _var_x->min() );
        }
     }
}//consistency

//! It checks if
bool
Bool2Int::satisfied () {
	
	// No variables: check the int values
    if ( _scope_size == 0 )
        return _arguments[ 0 ] == _arguments[ 1 ];

    // One variable: if its domain is empty, failed propagation
    if ( _scope_size == 1 && _var_x->is_empty() )
        return false;
    // One variable whose domain isn't empty: check if singleton
    if ( _scope_size == 1 && _var_x->is_singleton() )
        return _var_x->min() == _arguments[ 0 ];

    /*
     * Two variables: if the domain of either of them is empty,
     * failed propagation
     */
    if ( _scope_size == 2 &&
         (_var_x->is_empty() || _var_y->is_empty() ) )
        return false;
    // Two variables whose domain isn't empty: check if both are singletons
    if ( _scope_size == 2 &&
         _var_x->is_singleton() && _var_y->is_singleton() )
        return _var_x->min() == _var_y->min();

    /*
     * If there's not enough information to state whether
     * the constraint is satisfied or not, return true.
     */
    return true;
}//satisfied

//! Prints the semantic of this constraint
void
Bool2Int::print_semantic () const {
    BaseConstraint::print_semantic ();
    std::cout << "a = b\n";
    std::cout << "bool2int(var bool: a, var int:b)\n";
}//print_semantic



