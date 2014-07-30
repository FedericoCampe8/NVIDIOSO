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
}//IntVariable

IntVariable::IntVariable ( int idv ) :
Variable ( idv ) {
}//IntVariable

const DomainPtr
IntVariable::domain () {
  return _domain_ptr;
}//domain

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
}//shrink

bool
IntVariable::subtract ( int val ) {
  return _domain_ptr->subtract ( val );
}//subtract

void
IntVariable::in_min ( int min ) {
  _domain_ptr->in_min ( min );
}//in_min

void
IntVariable::in_max ( int max ) {
  _domain_ptr->in_max ( max );
}//in_max

void
IntVariable::print () const {
  Variable::print();
  _domain_ptr->print();
}//print

