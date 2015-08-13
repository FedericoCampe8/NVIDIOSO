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
	_global      ( false ),
	_number_id   ( -1 ),
	_str_id      ( "-1" ),
	_weight      ( 0 ),
	_consistency ( ConsistencyType::BOUND_C ),
	_shared_arguments ( nullptr ) {
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

bool
Constraint::is_naive () const 
{
	return (get_scope_size() == 0);
}//is_naive

bool
Constraint::is_unary () const 
{
	return (get_scope_size() == 1);
}//is_unary


bool
Constraint::is_global () const 
{
	return _global;
}//is_global

bool 
Constraint::is_soft () const
{
	return (get_weight () > 0);
}//is_soft

int
Constraint::get_weight () const 
{
  return _weight;
}//get_weight

void
Constraint::increase_weight ( int weight ) 
{
  _weight += weight;
}//increase_weight

void
Constraint::decrease_weight ( int weight ) {
  _weight -= weight;
}//decrease_weight

void
Constraint::set_consistency_level ( ConsistencyType con_type ) 
{
	_consistency = con_type;
}//set_consistency_level

void
Constraint::set_consistency_level ( std::string t ) 
{
	if ( t == "naive" )
	{
		_consistency = ConsistencyType::NAIVE_C;
	}
	else if ( t == "bound" )
	{
		_consistency = ConsistencyType::BOUND_C;
	}
	else if ( t == "domain" || t == "full" )
	{
		_consistency = ConsistencyType::DOMAIN_C;
	}
	else 
	{
		_consistency = ConsistencyType::NAIVE_C;
	}
}//set_consistency_level

ConsistencyType 
Constraint::get_consistency_level () const
{
	return _consistency;
}//set_propagator_type

void 
Constraint::set_var2subscript_mapping ( std::vector<int>& v )
{
	_var2subscription_map = v;
}//set_var2subscript_mapping

bool 
Constraint::is_variable_at ( int idx ) const
{
	assert ( idx >= 0 && idx < _var2subscription_map.size () );
	return _var2subscription_map [ idx ] == 1;
}//is_variable_at_index

size_t
Constraint::get_scope_size () const {
  return scope().size();
}//get_scope_size

size_t
Constraint::get_arguments_size () const {
  return _arguments.size ();
}//get_arguments_size

void
Constraint::set_event ( EventType event ) {
  _trigger_events.push_back ( event );
}//set_events

const std::vector<EventType>&
Constraint::events () const {
  return _trigger_events;
}//events

const std::vector<VariablePtr>
Constraint::scope () const 
{
  // Return the constraint's scope
  return _scope;
}//scope

void 
Constraint::set_shared_arguments ( std::unordered_map < std::string, std::vector<int> > * ptr )
{
	_shared_arguments = ptr;
}//set_shared_arguments

const std::vector<int>&
Constraint::get_shared_arguments ( size_t idx )
{
	// Sanity checks
	assert ( _shared_arguments != nullptr );
	assert ( _shared_argument_ids.size() > idx );
	
	auto it = _shared_arguments->find ( _shared_argument_ids [ idx ] );
	if ( it != _shared_arguments->end() )
	{
		return (it->second);
	}
	else
	{
		throw NvdException ( (_dbg + "get_shared_arguments: shared argument not found for id " + _shared_argument_ids [ idx ] ).c_str() );
	}
}//get_shared_arguments

const std::vector<int>&
Constraint::arguments () const {
  return _arguments;
}//arguments

void
Constraint::update ( EventType e ) {
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
Constraint::fix_point () const 
{
  return  ((changed_vars()).size() == 0);
}//fix_point

int
Constraint::unsat_level () const 
{
	return 0;
}//unsat_level



