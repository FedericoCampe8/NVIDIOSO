//
//  constraint.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "constraint.h"

using namespace std;

Constraint::Constraint () :
_number_id  ( -1 ),
_str_id     ( "-1" ),
_weight     ( 0 ) {
  _unique_id  = glb_id_gen->get_int_id();
  _dbg = ( "Constraint - " );
}//Constraint

Constraint::~Constraint () {
  _trigger_events.clear ();
  _arguments.clear ();
}//~Constraint

ConstraintPtr
Constraint::get_this_shared_ptr () {
  return shared_from_this();
}//make_shared

size_t
Constraint::get_unique_id () const {
  return _unique_id;
}//get_unique_id

int
Constraint::get_number_id () const {
  return _number_id;
}//get_number_id

std::string
Constraint::get_name () const {
  return _str_id;
}//get_name

int
Constraint::get_weight () const {
  return _weight;
}//get_weight

void
Constraint::increase_weight ( int weight ) {
  _weight += weight;
}//increase_weight

void
Constraint::decrease_weight ( int weight ) {
  _weight -= weight;
}//decrease_weight

size_t
Constraint::get_scope_size () const {
  return scope().size();
}//get_scope_size

size_t
Constraint::get_arguments_size () const {
  return _arguments.size ();
}//get_arguments_size

const std::vector<EventType>&
Constraint::events () const {
  return _trigger_events;
}//events

const std::vector<int>&
Constraint::arguments () const {
  return _arguments;
}//arguments

void
Constraint::update ( const Event& e ) {
}//update

std::vector<ConstraintPtr>
Constraint::decompose () const {
  assert( false );
  vector<ConstraintPtr> ptr_vec;
  return ptr_vec;
}//decompose

std::vector<VariablePtr>
Constraint::changed_vars_from_event ( EventType event ) const {

  vector<VariablePtr> var_array;
  for ( auto x : scope() ) {
    if ( x->get_event() == event ) {
      var_array.push_back ( x );
    }
  }
  
  return  var_array;
}//get_vars_from_event


std::vector<VariablePtr>
Constraint::changed_vars () const {
  
  vector<VariablePtr> var_array;
  
  for ( auto x : scope() ) {
    if ( (x->get_event() != EventType::NO_EVT) &&
         (x->get_event() != EventType::OTHER_EVT) ) {
      var_array.push_back ( x );
    }
  }
  
  return  var_array;
}//get_vars_from_event

bool
Constraint::fix_point () const {
  return  ((changed_vars()).size() == 0);
}//fix_point

int
Constraint::unsat_level () const {
  return 0;
}//unsat_level



