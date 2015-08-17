//
//  int_variable.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 29/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "int_variable.h"

IntVariable::IntVariable () :
Variable () {
  _domain_ptr       = nullptr;
  _backtack_manager = nullptr;
  set_backtrackable_id ();
}//IntVariable

IntVariable::IntVariable ( int idv ) :
Variable ( idv ) {
  _domain_ptr       = nullptr;
  _backtack_manager = nullptr;
  set_backtrackable_id ();
}//IntVariable

void
IntVariable::set_backtrack_manager ( BacktrackManagerPtr bkt_manager ) {
  _backtack_manager = bkt_manager;
}//set_backtrack_manager

void
IntVariable::notify_backtrack_manager () {
  if ( _backtack_manager == nullptr ) {
    throw NvdException ( (_dbg + "No Backtrack Manager to notify.").c_str() );
  }
  _backtack_manager->add_changed( get_backtrackable_id() );
}//notify_backtrack_manager

void
IntVariable::notify_observers () 
{
  	notify_backtrack_manager ();
  	notify_store ();
}//notify_observers

EventType
IntVariable::get_event () const {
  return _domain_ptr->get_event ();
}//get_event

void
IntVariable::reset_event () {
  _domain_ptr->reset_event ();
}//reset_event

void
IntVariable::set_domain_type ( DomainType  dt ) {
  _domain_ptr->set_type ( dt );
}//set_domain

size_t
IntVariable::get_size () const 
{
	return _domain_ptr->get_size();
}//get_size

bool
IntVariable::is_singleton () const {
  return _domain_ptr->is_singleton ();
}//is_singleton

bool
IntVariable::is_empty () const 
{
	return (get_size() == 0);
}//is_empty

int
IntVariable::min () const {
  return _domain_ptr->lower_bound();
}//min

int
IntVariable::max () const {
  return _domain_ptr->upper_bound();
}//max

void
IntVariable::shrink ( int min, int max ) 
{
  _domain_ptr->shrink ( min, max );
  notify_observers ();
}//shrink

bool
IntVariable::subtract ( int val ) 
{
	bool result = _domain_ptr->subtract ( val );
  	notify_observers ();
  	return result;
}//subtract

void
IntVariable::in_min ( int min ) {
  _domain_ptr->in_min ( min );
  notify_observers ();
}//in_min

void
IntVariable::in_max ( int max ) {
  _domain_ptr->in_max ( max );
  notify_observers ();
}//in_max

//! @note override backtrackable object methods.
void
IntVariable::set_backtrackable_id () {
  _backtrackable_id = _id;
}//set_id

void
IntVariable::print_domain() const {
  _domain_ptr->print ();
}//print_domain

void
IntVariable::print () const {
  Variable::print();
  _domain_ptr->print();
}//print

