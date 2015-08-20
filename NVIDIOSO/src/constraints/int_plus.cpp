//
//  int_plus.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Modified by Luca Foschiani on 08/14/15 (foschiani01@gmail.com).
//  Copyright (c) 2014-2015 Federico Campeotto. 
//	All rights reserved.
//

#include "int_plus.h"

IntPlus::IntPlus ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::INT_PLUS );
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  set_event( EventType::SINGLETON_EVT );
  set_event( EventType::MIN_EVT );
  set_event( EventType::MAX_EVT );
  set_event( EventType::BOUNDS_EVT );
}//IntPlus

void
IntPlus::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args ) 
{
	// Consistency checking in order to avoid more than one setup
    
	if ( (_var_x != nullptr) || (_arguments.size() > 0) ) return;

    
	if ( vars.size() == 0 ) 
	{// Consistency check on args
        
		if ( args.size() != 3 )
            
			throw NvdException ( (_dbg + "wrong number of arguments").c_str() );

        
		// Set arguments list
        
			_arguments.push_back( atoi ( args[ 0 ].c_str() ) );
        
			_arguments.push_back( atoi ( args[ 1 ].c_str() ) );
        
			_arguments.push_back( atoi ( args[ 2 ].c_str() ) );

        
		// Set scope size
        
		_scope_size = 0;
    
	}

	else if ( vars.size() == 1 ) 
	{

// Consistency check on args

		if ( args.size() != 2 )

			throw NvdException ( (_dbg + "wrong number of arguments").c_str() );


			// Set variables

			_var_x =
 std::dynamic_pointer_cast<IntVariable>( vars[ 0 ]);


			// Consistency check on pointers

			if ( _var_x == nullptr )

				throw NvdException ( (_dbg + "x variable is NULL").c_str() );


			// Set arguments list

			_arguments.push_back( atoi ( args[ 0 ].c_str() ) );

			_arguments.push_back( atoi ( args[ 1 ].c_str() ) );


			// Set scope size

			_scope_size = 1;

	}
	else if ( vars.size() == 2 ) 
	{// Consistency check on args
		if ( args.size() != 1 )

			throw NvdException ( (_dbg + "wrong number of arguments").c_str() );


		// Set variables

		_var_x =
 std::dynamic_pointer_cast<IntVariable>( vars[ 0 ]);
		_var_y =
 std::dynamic_pointer_cast<IntVariable>( vars[ 1 ]);


		// Consistency check on pointers

		if ( _var_x == nullptr )

			throw NvdException ( (_dbg + "x variable is NULL").c_str() );
		if ( _var_y == nullptr )
            
			throw NvdException ( (_dbg + "y variable is NULL").c_str() );


		// Set arguments list

		_arguments.push_back( atoi ( args[ 0 ].c_str() ) );


		// Set scope size

		_scope_size = 2;

	}
	else if ( vars.size() == 3 )
 
	{
// Set variables

		_var_x =
 std::dynamic_pointer_cast<IntVariable>( vars[ 0 ]);


		_var_y =
 std::dynamic_pointer_cast<IntVariable>( vars[ 1 ]);


		_var_z =
 std::dynamic_pointer_cast<IntVariable>( vars[ 2 ]);


		// Consistency check on pointers

		if ( _var_x == nullptr )
        	throw NvdException ( (_dbg + "x variable is NULL").c_str() );

      	if ( _var_y == nullptr )
        	throw NvdException ( (_dbg + "y variable is NULL").c_str() );
     	if ( _var_z == nullptr )
         	throw NvdException ( (_dbg + "z variable is NULL").c_str() );

       	// Set scope size

		_scope_size = 3;

	}
}//setup


const std::vector<VariablePtr>
IntPlus::scope () const 
{// Return the constraint's scope
  std::vector<VariablePtr> scope;
  if ( _var_x != nullptr ) scope.push_back ( _var_x );
  if ( _var_y != nullptr ) scope.push_back ( _var_y );
  if ( _var_z != nullptr ) scope.push_back ( _var_z );
  return scope;
}//scope

void
IntPlus::consistency () 
{
	

// No variables: no propagations

	if ( _scope_size == 0 ) return;

	/*
   	 * One variable not singleton: propagate on it
   	 * after checking if the variable is one of the

   	 * summands or the result
   	 */
   	else if ( _scope_size == 1 &&
 !_var_x->is_singleton() )
   	{


   		if ( is_variable_at( 0 ) ||
 is_variable_at( 1 )) 
   		{

   			int bound = _arguments[ 1 ] - _arguments[ 0 ];
   			_var_x->shrink( bound, bound );

   		}
   		else
   		{

   			int bound = _arguments[ 0 ] + _arguments[ 1 ];

   			_var_x->shrink( bound, bound );

   		}


   		return;

   	}


   	/*
   	 * Two variables:
   	 * first check if the variables are

   	 * both summands or one is the result,

   	 * then propagate on variables that

   	 * aren't singletons
   	 */

   	 else if ( _scope_size == 2 )
   	 {


   	 	if ( !is_variable_at( 2 ) )
   	 	{
   	 		

if ( !_var_x->is_singleton() )
   	 		{
   	 			int min_x = _arguments[ 0 ] - _var_y->max();

   	 			int max_x = _arguments[ 0 ] - _var_y->min();

   	 			_var_x->in_min ( min_x );

   	 			_var_x->in_max ( max_x );
   	 		}


   	 		if ( !_var_y->is_singleton() )
   	 		{
   	 			int min_y = _arguments[ 0 ] - _var_x->max();

   	 			int max_y = _arguments[ 0 ] - _var_x->min();

   	 			_var_y->in_min ( min_y );

   	 			_var_y->in_max ( max_y );

   	 		}


   	 	}
   	 	else
   	 	{
   	 		if ( !_var_x->is_singleton() )
   	 		{
   	 			
int min_x = _var_y->min() - _arguments[ 0 ];
   	 			int max_x = _var_y->max() - _arguments[ 0 ];

   	 			_var_x->in_min ( min_x );

   	 			_var_x->in_max ( max_x );
   	 		}


   	 		if ( !_var_y->is_singleton() )
   	 		{
   	 			
int min_y = _arguments[ 0 ] + _var_x->min();
   	 			int max_y = _arguments[ 0 ] + _var_x->max();
   	 			_var_y->in_min ( min_y );
   	 			_var_y->in_max ( max_y );
   	 		}

   	 	}


   	 	return;

   	 }


   	 /*

   	  * Three variables:

   	  * propagate on all variables that aren't

   	  * singletons

   	  */

   	  else if ( _scope_size == 3 )
   	  {
   	  	

if ( !_var_x->is_singleton() )
   	  	{
   	  		
int min_x = _var_z->min() - _var_y->max();

   	  		int max_x = _var_z->max() - _var_y->min();
   	  		_var_x->in_min ( min_x );
   	  		_var_x->in_max ( max_x );

   	  	}


   	  	if ( !_var_y->is_singleton() )
   	  	{

   	  		int min_y = _var_z->min() - _var_x->max();

   	  		int max_y = _var_z->max() - _var_x->min();
   	  		_var_y->in_min ( min_y );
   	  		_var_y->in_max ( max_y );
   	  	}
   	  	if ( !_var_z->is_singleton() )
   	  	{
   	  		int min_z = _var_x->min() + _var_y->min();
   	  		int max_z = _var_x->max() + _var_y->max();

   	  		_var_z->in_min ( min_z );

   	  		_var_z->in_max ( max_z );
   	  	}


   	  	return;

   	  }
}//consistency

//! It checks if x + y = z
bool
IntPlus::satisfied () 
{

    
	// No variables: check the int values
    
	if ( _scope_size == 0 ) 
	{
        
		return _arguments[ 0 ] + _arguments[ 1 ] == _arguments[ 2 ];
    
	}

    
	// One variable: if its domain is empty, failed propagation
    
	if ( _scope_size == 1 &&
 _var_x->is_empty() )
        
		return false;
    
		
	// One variable whose domain isn't empty: check if singleton
    
	if ( _scope_size == 1 &&
 _var_x->is_singleton() ) 
	{
        
		if ( is_variable_at( 0 ) ||
 is_variable_at( 1 ) ) 
		{
            
			return _var_x->min() + _arguments[ 0 ] == _arguments[ 1 ];
        
		} 
		else 
		{
            
			return _arguments[ 0 ] + _arguments[ 1 ] == _var_x->min();
        
		}
    
	}


	    
	/*
     
	 * Two variables: if the domain of either of them is empty,
     
	 * failed propagation
     
	 */
    
	 if ( _scope_size == 2 &&
 (_var_x->is_empty() ||
 _var_y->is_empty() ) )
		return false;
    
		
	// Two variables whose domain isn't empty: check if both are singletons
    
	if ( _scope_size == 2 &&
 _var_x->is_singleton() &&
 _var_y->is_singleton() ) 
	{
        
		if ( !is_variable_at( 2 ) ) 
		{
            
			return _var_x->min() + _var_y->min() == _arguments[ 0 ];
        
		} 
		else 
		{
            
			return _arguments[ 0 ] + _var_x->min() == _var_y->min();
  
		}
    
	}

    
	
	/*
     
	 * Three variables: if the domain of at least one
     
	 * of them is empty, failed propagation
    
	 */
    
	 if ( _scope_size == 3 &&
 (_var_x->is_empty() ||
 _var_x->is_empty() ||
 _var_x->is_empty() ) )

		return false;
    
		
	// Three variables whose domain isn't empty: check if all are singletons
    
	if ( _scope_size == 3 &&
 _var_x->is_singleton () &&
 _var_y->is_singleton () &&
 _var_z->is_singleton () ) 
	{
        
		return _var_x->min() + _var_y->min() == _var_z->min();
    
	}
	

    
	/*
     
	 * If there's not enough information to state whether
     
	 * the constraint is satisfied or not, return true.
     
	 */
    
	 return true;
}//satisfied

//! Prints the semantic of this constraint
void
IntPlus::print_semantic () const 
{
	BaseConstraint::print_semantic ();

	std::cout << "a + b = c\n";

	std::cout << "int_plus(var int: a, var int: b, var int: c)\n";
}//print_semantic