//
//  array_bool_element.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "array_bool_element.h"

ArrayBoolElement::ArrayBoolElement ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::ARRAY_BOOL_ELEMENT );
	_ground_var_int = false;
	
  	/*
     * Set the event that trigger this constraint.
   	 * @note if no event is set, this constraint will never be re-evaluated.
   	 */
  	set_event( EventType::SINGLETON_EVT );
}//ArrayBoolElement

ArrayBoolElement::~ArrayBoolElement () {}

void
ArrayBoolElement::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
	if ( vars.size() == 0 ) 
  	{  
    	// Sanity check
    	if ( args.size() != 3 )
    	{
      		throw NvdException ( (_dbg + "wrong number of arguments").c_str() );
    	}
    	
    	// Set scope size (default: 0) ang arguments list
    	_arguments.push_back( atoi ( args[ 0 ].c_str() ) );
    	_arguments.push_back( atoi ( args[ 2 ].c_str() ) );
    	_shared_argument_ids = std::vector<std::string> ( 1, args[ 1 ] );
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
    	if ( args.size() != 2 )
    	{
    		throw NvdException ( (_dbg + "wrong number of arguments").c_str() );
    	}
    
    	// Set scope size ang arguments list
    	_scope_size = 1;
    	
    	/*
    	 * @note To distinguish the ground variable from the name of the array of 
    	 *       shared elements here we parse the string. If one
    	 *       of the chars of args[i] is not in [0..9], then args[i] is the
    	 *       name of the shared array of elements.
    	 * @todo Use a function to identify variables, avoid this check.
    	 */
    	 bool found = false;
    	 for ( auto& c : args[ 0 ] )
    	 {
    	 	if ( ((int) c) < '0' || ((int) c) > '9' )
    	 	{
    	 		_arguments.push_back( atoi ( args[ 1 ].c_str() ) );
    			_shared_argument_ids = std::vector<std::string> ( 1, args[ 0 ] );
    			found = true;
    			
    			// var bool is ground
    			_ground_var_int = false;
    			break;
    	 	}
    	 }
    	 if ( !found )
    	 {
    	 	for ( auto& c : args[ 1 ] )
    	 	{
    	 		if ( ((int) c) < '0' || ((int) c) > '9' )
    	 		{
    	 			_arguments.push_back( atoi ( args[ 0 ].c_str() ) );
    				_shared_argument_ids = std::vector<std::string> ( 1, args[ 1 ] );
    				found = true;
    				
    				// var int is ground
    				_ground_var_int = true;
    				break;
    	 		}
    	 	}
    	 }
    	// Sanity check
    	if ( !found )
    	{
    		throw NvdException ( (_dbg + "ground argument not found").c_str() );
    	}
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
		
		// Sanity check on args
    	if ( args.size() != 1 )
    		throw NvdException ( (_dbg + "wrong number of arguments").c_str() );
    	
    	// Set scope size ang arguments list
    	_scope_size = 2;
    	_shared_argument_ids = args;
  	}
}//setup

const std::vector<VariablePtr>
ArrayBoolElement::scope () const
{
  // Return constraint's scope
  std::vector<VariablePtr> scope;
  if ( _var_x != nullptr ) scope.push_back ( _var_x );
  if ( _var_y != nullptr ) scope.push_back ( _var_y );
  return scope;
}//scope

void
ArrayBoolElement::consistency ()
{
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
  	if ( _scope_size == 1 ) 
  	{
    	if ( !_var_x->is_singleton() )
    	{
    		if ( _ground_var_int )
    		{// Propagate on var_bool -> var_bool = as[var_int]
    		
    			/*
    			 * 				========= NOTE =========
    			 * 		get_shared_arguments()[ 2 + _arguments[ 0 ] - 1 ]
    			 *  Where get_shared_arguments() by default is get_shared_arguments( 0 ),
    			 *  _arguments[ 0 ] is the array index 1-based 
    			 *  AND
    			 * the array starts at position 2 since the first 2 array's elements 
    			 * (0 and 1) are used to store number of row and number of columns.
    			 *				========================
    			 */ 
    			_var_x->subtract( 1 - (get_shared_arguments()[ 2 + (_arguments[ 0 ] - 1) ]) );
    		}	
    		else
    		{// Propagate on var_int -> var_int = all vals i s.t. as[i] == var_bool
    			
    			// @note starting from index 2
    			for ( int i = 2; i < (get_shared_arguments()).size(); i++ )
    			{
    				if ( get_shared_arguments()[i] != _arguments[ 0 ] )
    				{
    					_var_x->subtract( i );
    				}
    			}
      		}
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
      		_var_y->subtract( 1 - get_shared_arguments()[ 2 + (_var_x->min () - 1) ] );
    	}
    	else if ( (_var_y->is_singleton()) && (!_var_x->is_singleton()) ) 
    	{
    		int value = _var_y->min ();
    		for ( int i = 2; i < (get_shared_arguments()).size(); i++ )
    		{
    			if ( get_shared_arguments()[i] != value )
    			{
    				_var_x->subtract( i );
    			}
    		}
    	}
    	return;
  	}
}//consistency

//! It checks if as[b] = c
bool
ArrayBoolElement::satisfied ()
{
	// No FD variables
  	if ( _scope_size == 0 ) 
  	{
    	return (_arguments[ 1 ] == get_shared_arguments()[ 2 + (_arguments[ 0 ] - 1) ]);
  	}
  	
  	// 1 FD variable, if singleton check
  	if ( (_scope_size == 1) && _ground_var_int && _var_x->is_singleton() ) 
	{
    	return (_var_x->min () == get_shared_arguments()[ 2 + (_arguments[ 0 ] - 1) ]);
  	}
  	
  	// 1 FD variable, if singleton check
  	if ( (_scope_size == 1) && !_ground_var_int && _var_x->is_singleton() ) 
	{
		return (_arguments[ 0 ] == get_shared_arguments()[ 2 + (_var_x->min () - 1) ]);
  	}
  
  	// 2 FD variables, if singleton check
  	if ( (_scope_size == 2) &&
       	 (_var_x->is_singleton()) &&
       	 (_var_y->is_singleton()) ) 
    {
    	return (_var_y->min () == get_shared_arguments()[ 2 + (_var_x->min () - 1) ]);
  	}
  
	/*
   	 * Check if a domain is empty.
   	 * If it is the case: failed propagation.
   	 */
   	if ( _scope_size == 1 && _var_x->is_empty () )
	{
    	return false;
    }
    
  	if ( (_scope_size == 2) && 
       	 (_var_x->is_empty () || _var_y->is_empty () ) )
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
ArrayBoolElement::print_semantic () const
{
    BaseConstraint::print_semantic ();
    std::cout << "b in [1..n], as[b] = c\n";
  	std::cout << "array_bool_element(var int: b, array [int] of bool: as, var bool: c)\n";
}//print_semantic



