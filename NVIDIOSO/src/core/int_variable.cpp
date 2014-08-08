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
  domain_iterator = new DomainIterator( _domain_ptr );
}//IntVariable

IntVariable::IntVariable ( int idv ) :
Variable ( idv ) {
  domain_iterator = new DomainIterator( _domain_ptr );
}//IntVariable

EventType
IntVariable::get_event () const {
  return _domain_ptr->get_event ();
}//get_event

void
IntVariable::set_domain_type ( DomainType  dt ) {
  _domain_ptr->set_type ( dt );
}//set_domain

size_t
IntVariable::get_size () const {
  return _domain_ptr->get_size();
}//get_size

bool
IntVariable::is_singleton () const {
  return _domain_ptr->is_singleton ();
}//is_singleton

bool
IntVariable::is_empty () const {
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
IntVariable::shrink ( int min, int max ) {
  _domain_ptr->shrink ( min, max );
  notify_store ();
}//shrink

bool
IntVariable::subtract ( int val ) {
  bool result = _domain_ptr->subtract ( val );
  notify_store ();
  return result;
}//subtract

void
IntVariable::in_min ( int min ) {
  _domain_ptr->in_min ( min );
  notify_store ();
}//in_min

void
IntVariable::in_max ( int max ) {
  _domain_ptr->in_max ( max );
  notify_store ();
}//in_max

void
IntVariable::print () const {
  Variable::print();
  _domain_ptr->print();
}//print

