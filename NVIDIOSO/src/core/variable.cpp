//
//  cp_variable.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "variable.h"
#include "constraint.h"
#include "constraint_store.h"

using namespace std;

Variable::Variable () :
_str_id              ( "" ),
_constraint_store    ( nullptr ),
_var_type            ( VariableType::OTHER ),
_number_of_constraints ( 0 ) {
  _id = glb_id_gen->get_int_id();
}//Variable

Variable::Variable ( int v_id ) :
_id                  ( v_id ),
_str_id              ( "" ),
_constraint_store    ( nullptr ),
_var_type            ( VariableType::OTHER ),
_number_of_constraints ( 0 ) {
}//Variable

Variable::~Variable () {
  _attached_constraints.clear();
  _detach_constraints.clear();
  delete domain_iterator;
}//~Variable

int
Variable::get_id () const {
  return _id;
}//get_id

void
Variable::set_str_id ( string str_id ) {
  if ( _str_id.compare( "" ) == 0 ) {
    _str_id = str_id;
  }
}//set_str_id

string
Variable::get_str_id () const {
  return _str_id;
}//get_str_id

void
Variable::set_type ( VariableType v_type ) {
  if ( _var_type == VariableType::OTHER ) {
    _var_type = v_type;
  }
}//set_type

VariableType
Variable::get_type () const {
  return _var_type;
}//get_type

bool
Variable::is_empty () const {
  return ( get_size() == 0 );
}//is_empty

void
Variable::attach_store ( ConstraintStorePtr store ) {
  if ( store == nullptr ) {
    throw NvdException ( (_dbg + "Trying to set a null pointer as store").c_str() );
  }
  _constraint_store = store;
}//attach_store

void
Variable::attach_constraint ( ConstraintPtr c ) {
  
  // Consistency check
  if ( c == nullptr ) return;
  
  /*
   * Check if c is the list of detached constraints.
   * If it is the case, move from detached constraints
   * to attached_constraints. 
   * Otherwise attach c to the current
   * list of attached constraints and increase size.
   */
  auto iter = _detach_constraints.begin();
  while ( iter != _detach_constraints.end() ) {
    if ( *iter == c->get_unique_id() ) {
      _detach_constraints.erase( iter );
      return;
    }
  }
  
  /*
   * Otherwise add c to the observers and increase 
   * the total number of observers.
   */
  for ( auto event : c->events() ) {
    if ( _attached_constraints.find( event ) == _attached_constraints.end() ) {
      _attached_constraints[ event ].push_back ( c );
    }
    else {
      for ( int i = 0; i < _attached_constraints[ event ].size(); i++ ) {
        if ( _attached_constraints[ event ][ i ]->get_unique_id() ==
             c->get_unique_id ()) {
          return;
        }
      }//i
      _attached_constraints[ event ].push_back ( c );
    }
  }
  
  _number_of_constraints++;
}//attach_constraint

void
Variable::notify_constraint () {
  EventType event = get_event();
  bool find;
  for ( auto c : _attached_constraints[ event ] ) {
    find = false;
    for ( auto det : _detach_constraints ) {
      if ( det == c->get_unique_id() ) {
        find = true;
        break;
      }
    }
    if ( !find ) {
      c->update ( event );
    }
  }
}//notify_constraint

void
Variable::notify_store () {

  if ( _constraint_store == nullptr ) {
    throw NvdException ( (_dbg + "No store attached to this constraint").c_str() );
  }
  
  EventType event = get_event ();
  
  /*
   * If there is a fail event, then
   * notify store to stop re-evaluating constraints
   * as soon as possible.
   */
  if ( event == EventType::FAIL_EVT ) {
    _constraint_store->fail ();
    return;
  }
  
  vector< size_t > constraints_to_reevaluate;
  for ( auto c : _attached_constraints[ event ] ){
    if ( is_attached( c->get_unique_id() ) ) {
      constraints_to_reevaluate.push_back( c->get_unique_id() );
    }
  }
  
  // Update constraint store
  _constraint_store->add_changed ( constraints_to_reevaluate, event );
  constraints_to_reevaluate.clear ();
}//notify_store

void
Variable::detach_constraint ( ConstraintPtr c ) {
  // Consistency check
  if ( c == nullptr ) return;
  detach_constraint ( c->get_unique_id() );
}//detach_constraint

void
Variable::detach_constraint ( size_t c_id ) {
  _detach_constraints.push_back( c_id );
}//detach_constraint

size_t
Variable::size_constraints () {
  if ( _attached_constraints.size() == 0 ) return 0;
  
  size_t not_sat = 0;
  map < size_t, Constraint * > valid_constraints;
  for ( auto x : _attached_constraints ) {
    for ( auto y : x.second ) {
      if ( is_attached (y->get_unique_id()) ) {
        valid_constraints[ y->get_unique_id() ] = y.get();
      }
    }
  }
  
  if ( valid_constraints.size() == 0 ) return 0;
  
  for ( auto x : valid_constraints ) {
    if ( !x.second->satisfied() ) not_sat++;
  }
  
  return not_sat;
}//size_constraints

bool
Variable::is_attached ( size_t c_id ) {
  if ( _detach_constraints.size() == 0 ) return true;
  
  for ( auto x : _detach_constraints )
    if ( x == c_id ) return false;
  return true;
}//is_attached

size_t
Variable::size_constraints_original () const {
  return _number_of_constraints;
}//size_constraints_original

void
Variable::print () const  {
  cout << "Variable_"   << _id     << "\n";
  cout << "Variable id: \"" << _str_id << "\"\n";
  cout << "Type:\t";
  switch ( _var_type ) {
    case VariableType::FD_VARIABLE:
      cout << "FD_Var\n";
      break;
    case VariableType::SUP_VARIABLE:
      cout << "SUP_Var\n";
      break;
    case VariableType::OBJ_VARIABLE:
      cout << "OBJ_Var\n";
      break;
    default:
      cout << "Not Defined\n";
  }
}//print



