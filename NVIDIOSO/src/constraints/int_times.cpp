//
//  int_times.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Modified by Luca Foschiani on 08/18/15 (foschiani01@gmail.com).
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//


#include "int_times.h"

IntTimes::IntTimes () :
FZNConstraint ( INT_TIMES ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  set_event( EventType::SINGLETON_EVT );

  set_event( EventType::MIN_EVT );

  set_event( EventType::MAX_EVT );

  set_event( EventType::BOUNDS_EVT );
}//IntTimes

IntTimes::IntTimes ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
IntTimes () {
  setup ( vars, args );
}//IntTimes

void
IntTimes::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args ) {


    // Consistency checking in order to avoid more than one setup
    if ( (_var_x != nullptr) || (_arguments.size() > 0) ) return;

    if ( vars.size() == 0 ) {

        // Consistency check on args
        if ( args.size() != 3 )
            throw NvdException ( (_dbg + "wrong number of arguments").c_str() );

        // Set arguments list
        _arguments.push_back( atoi ( args[ 0 ].c_str() ) );
        _arguments.push_back( atoi ( args[ 1 ].c_str() ) );
        _arguments.push_back( atoi ( args[ 2 ].c_str() ) );

        // Set scope size
        _scope_size = 0;
    }
    else if ( vars.size() == 1 ) {

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
    else if ( vars.size() == 2 ) {

        // Consistency check on args
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
IntTimes::scope () const {

  // Return the constraint's scope

  std::vector<VariablePtr> scope;

  if ( _var_x != nullptr ) scope.push_back ( _var_x );

  if ( _var_y != nullptr ) scope.push_back ( _var_y );

  if ( _var_z != nullptr ) scope.push_back ( _var_z );

  return scope;

}//scope

void
IntTimes::consistency () {

    // No variables: no propagations
    if ( _scope_size == 0 ) {
        return;
    }

    /*

   	 * One variable not singleton: propagate on it

   	 * after checking if the variable is one of the
   	 * factors or the result

   	 */
    else if ( _scope_size == 1 &&
              !_var_x->is_singleton() ) {

        if ( is_variable_at( 0 ) ||
             is_variable_at( 1 )) {
            if ( _arguments[ 0 ] != 0) {
                int bound = _arguments[ 1 ] / _arguments[ 0 ];
                _var_x->shrink( bound, bound );
            }
        } else {
            int bound = _arguments[ 0 ] * _arguments[ 1 ];
            _var_x->shrink( bound, bound );
        }

        return;
    }

    /*

   	 * Two variables:

   	 * first check if the variables are
   	 * both factors or one is the result,
     * then propagate on variables that
     * aren't singletons

   	 */
    else if ( _scope_size == 2 ) {

        if ( !is_variable_at( 2 ) ) {

            if ( !_var_x->is_singleton() ) {

                std::pair<int,int> bounds = div_bounds ( _arguments[0], _arguments[ 0 ], _var_y->min(), _var_y->max() );

                _var_x->in_min ( bounds.first );
                _var_x->in_max ( bounds.second );
            }

            if ( !_var_y->is_singleton() ) {

                std::pair<int,int> bounds = div_bounds ( _arguments[0], _arguments[ 0 ], _var_x->min(), _var_x->max() );

                _var_y->in_min ( bounds.first );
                _var_y->in_max ( bounds.second );
            }

        } 
        else 
        {

            if ( !_var_x->is_singleton() ) {

                std::pair<int,int> bounds = div_bounds ( _var_y->min(), _var_y->max(), _arguments[0], _arguments[ 0 ] );

                _var_x->in_min ( bounds.first );
                _var_x->in_max ( bounds.second );
            }

            if ( !_var_y->is_singleton() ) {

                std::pair<int,int> bounds = mul_bounds ( _arguments[0], _arguments[ 0 ], _var_x->min(), _var_x->max() );

                _var_y->in_min ( bounds.first );
                _var_y->in_max ( bounds.second );
            }
        }

        return;
    }

    /*
     * Three variables:
     * propagate on all variables that aren't
     * singletons
     */
     else if ( _scope_size == 3 ) {

        if ( !_var_x->is_singleton() ) {

            std::pair<int,int> bounds = div_bounds ( _var_z->min(), _var_z->max(), _var_y->min(), _var_y->max() );

            _var_x->in_min ( bounds.first );
            _var_x->in_max ( bounds.second );
        }

        if ( !_var_y->is_singleton() ) {

            std::pair<int,int> bounds = div_bounds ( _var_z->min(), _var_z->max(), _var_x->min(), _var_x->max() );

            _var_y->in_min ( bounds.first );
            _var_y->in_max ( bounds.second );
        }

        if ( !_var_z->is_singleton() ) {

            std::pair<int,int> bounds = mul_bounds ( _var_x->min(), _var_x->max(), _var_y->min(), _var_y->max() );

            _var_z->in_min ( bounds.first );
            _var_z->in_max ( bounds.second );
        }
    }

    return;
}//consistency

/*
 * Returns the bounds of the domain
 * [d1..d2]*[e1..e2]
 */
std::pair<int,int>
IntTimes::mul_bounds ( int d1, int d2, int e1, int e2 ) {

    int l_bound,u_bound;
    if ( d1 >= 0 ) {
        if ( e1 >= 0 ) {
            l_bound = d1 * e1;
            u_bound = d2 * e2;
        } else if ( e2 <= 0 ) {
            l_bound = d2 * e1;
            u_bound = d1 * e2;
        } else {
            l_bound = d2 * e1;
            u_bound = d2 * e2;
        }
    } else if ( d2 <= 0 ) {
        if ( e1 >= 0 ) {
            l_bound = d1 * e2;
            u_bound = d2 * e1;
        } else if ( e2 <= 0 ) {
            l_bound = d2 * e2;
            u_bound = d1 * e1;
        } else {
            l_bound = d1 * e2;
            u_bound = d1 * e1;
        }
    } else {
        if ( e1 >= 0 ) {
            l_bound = d1 * e2;
            u_bound = d2 * e2;
        } else if ( e2 <= 0 ) {
            l_bound = d2 * e1;
            u_bound = d1 * e1;
        } else {
            l_bound = std::min( d1*e2, d2*e1 );
            u_bound = std::max( d1*e1, d2*e2 );
        }
    }

    return std::make_pair ( l_bound, u_bound );
}//mul_bounds

/*
 * Returns the bounds of the domain
 * [d1..d2]/[e1..e2]
 */
std::pair<int,int>
IntTimes::div_bounds ( int d1, int d2, int e1, int e2 ) {

    int l_bound,u_bound;
    if ( d1 >= 0 ) {
        if ( e1 >= 0 ) {
            if ( e2 != 0 )
                l_bound = d1 / e2;
            if ( e1 !=  0 )
                u_bound = d2 / e1;
            else if ( e2 > 0 )
                u_bound = d2 / 1;
        } else if ( e2 <= 0 ){
            if ( e2 != 0)
                l_bound = d2 / e2;
            else if ( e1 < 0 )
                l_bound = d2 / -1;
            if ( e1 != 0 )
                u_bound = d1 / e1;
        } else {
            l_bound = d2 / -1;
            u_bound = d2 / 1;
        }
    } else if ( d2 <= 0 ) {
        if ( e1 >= 0 ) {
            if ( e1 != 0 )
                l_bound = d1 / e1;
            else if ( e2 > 0 )
                l_bound = d1 / 1;
            if ( e2 != 0 )
                u_bound = d2 / e2;
        } else if ( e2 <= 0 ) {
            if ( e1 != 0 )
                l_bound = d2 / e1;
            if ( e2 != 0 )
                u_bound = d1 / e2;
            else if ( e1 < 0 )
                u_bound = d1 / -1;
        } else {
            l_bound = d1 / 1;
            u_bound = d1 / -1;
        }
    } else {
        if ( e1 >= 0 ) {
            if ( e1 != 0 )
                l_bound = d1 / e1;
            else if ( e2 > 0 )
                l_bound = d1 / 1;
            if ( e1 != 0 )
                u_bound = d2 / e1;
            else if ( e2 > 0 )
                u_bound = d2 / 1;
        } else if ( e2 <= 0 ) {
            if ( e2 != 0 )
                l_bound = d2 / e2;
            else if ( e1 < 0 )
                l_bound = d2 / -1;
            if ( e2 != 0 )
                u_bound = d1 / e2;
            else if ( e1 < 0 )
                u_bound = d1 / -1;
        } else {
            l_bound = std::min( d1/e2, d2/e1 );
            u_bound = std::max( d1/e1, d2/e2 );
        }
    }

    return std::make_pair ( l_bound, u_bound );
}//div_bounds

//! It checks if x * y = z
bool
IntTimes::satisfied () 
{
    // No variables: check the int values
    if ( _scope_size == 0 ) 
    {
        return _arguments[ 0 ] * _arguments[ 1 ] == _arguments[ 2 ];
    }

    // One variable: if its domain is empty, failed propagation
    if ( _scope_size == 1 && _var_x->is_empty() )
        return false;
        
    // One variable whose domain isn't empty: check if singleton
    if ( _scope_size == 1 && _var_x->is_singleton() ) 
    {
        if ( is_variable_at( 0 )|| is_variable_at( 1 ) ) 
        {
            return _var_x->min() * _arguments[ 0 ] == _arguments[ 1 ];
        } 
        else 
        {
            return _arguments[ 0 ] * _arguments[ 1 ] == _var_x->min();
        }
    }
	
    /*
     * Two variables: if the domain of either of them is empty,
     * failed propagation
     */
    if ( _scope_size == 2 && (_var_x->is_empty() || _var_y->is_empty() ) )
        return false;
        
    // Two variables whose domain isn't empty: check if both are singletons
    if ( _scope_size == 2 && _var_x->is_singleton() && _var_y->is_singleton() ) 
    {
        if ( !is_variable_at( 2 ) ) 
        {
            return _var_x->min() * _var_y->min() == _arguments[ 0 ];
        } 
        else 
        {
            return _arguments[ 0 ] * _var_x->min() == _var_y->min();
        }
    }

    /*
     * Three variables: if the domain of at least one of them is empty,
     * failed propagation
     */
    if ( _scope_size == 3 &&
         (_var_x->is_empty() ||
          _var_y->is_empty() ||
          _var_z->is_empty() ) )
        return false;
        
    // Three variables whose domain isn't empty: check if all are singletons
    if ( _scope_size == 3 &&
         _var_x->is_singleton() &&
         _var_y->is_singleton() &&
         _var_z->is_singleton() ) {
        return _var_x->min() * _var_y->min() == _var_z->min();
    }
	
    /*
     * If there's not enough information to state whether
     * the constraint is satisfied or not, return true.
     */
    return true;

}//satisfied

//! Prints the semantic of this constraint
void
IntTimes::print_semantic () const
{
    FZNConstraint::print_semantic ();
    std::cout << "a * b = c\n";
    std::cout << "int_times(var int: a, var int: b, var int: c)\n";
}//print_semantic
