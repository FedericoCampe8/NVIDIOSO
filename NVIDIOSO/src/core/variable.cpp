//
//  cp_variable.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "variable.h"
#include "constraint.h"

using namespace std;

Variable::Variable () :
_str_id              ( "" ),
_var_type            ( VariableType::OTHER ),
_number_of_observers ( 0 ) {
  _id = glb_id_gen->get_int_id();
}//Variable

Variable::Variable ( int id ) :
_id                  ( id ),
_str_id              ( "" ),
_var_type            ( VariableType::OTHER ),
_number_of_observers ( 0 ) {
}//Variable

Variable::~Variable () {
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
Variable::attach_constraint ( ObserverPtr c ) {
  
  // Consistency check
  if ( c == nullptr ) return;
  
  /*
   * Check if c is the list of detached observers.
   * If it is the case, move from detached observers
   * to observers. Otherwise attach c to the current
   * list of observers and increase size.
   */
  auto iter = _detach_observers.begin();
  while ( iter != _detach_observers.end() ) {
    if ( *iter == c->get_unique_id() ) {
      _observers.push_back( c );
      _detach_observers.erase( iter );
      return;
    }
  }
  
  /*
   * Otherwise add c to the observers and increase 
   * the total number of observers.
   */
  _observers.push_back( c );
  _number_of_observers++;
}//attach_constraint

void
Variable::notify_constraint () {
  Event event ( get_event() );
  for ( auto c : _observers )
    c->update ( event );
}//notify_constraint

void
Variable::notify_store () {
}//notify_store

void
Variable::detach_constraint ( ObserverPtr c ) {
  
  // Consistency check
  if ( c == nullptr ) return;
  
  /*
   * Check if c is among the list of observers.
   * If it is the case, detach it, otherwise return.
   */
  auto iter = _observers.begin();
  while ( iter != _observers.end() ) {
    if ( (*iter)->get_unique_id() == c->get_unique_id() ) {
      _detach_observers.push_back( c->get_unique_id() );
      _observers.erase( iter );
      return;
    }
  }
}//detach_constraint

void
Variable::detach_constraint ( size_t c_id ) {
  
  /*
   * Check if c is among the list of observers.
   * If it is the case, detach it, otherwise return.
   */
  auto iter = _observers.begin();
  while ( iter != _observers.end() ) {
    if ( (*iter)->get_unique_id() == c_id ) {
      _detach_observers.push_back( c_id );
      _observers.erase( iter );
      return;
    }
  }
}//detach_constraint

size_t
Variable::size_constraints () {
  size_t not_sat = 0;
  for ( auto x : _observers ) {
    if ( !x->satisfied() ) not_sat++;
  }
  
  return not_sat;
}//size_constraints

size_t
Variable::size_constraints_original () const {
  return _number_of_observers;
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



