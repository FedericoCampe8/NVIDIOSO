//
//  set_domain.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 09/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "set_domain.h"

using namespace std;

SetDomain::SetDomain () {
  _dbg = "SetDomain - ";
  _dom_type = DomainType::SET;
}//SetDomain

SetDomain::~SetDomain () {
}//~SetDomain

DomainPtr
SetDomain::clone () const {
  return clone_impl ();
}//clone

DomainPtr
SetDomain::clone_impl () const {
  return ( shared_ptr<SetDomain> ( new SetDomain ( *this ) ) );
}//clone_impl

EventType
SetDomain::get_event () const {
  return EventType::OTHER_EVT;
}//get_event

void
SetDomain::set_values( vector<int> values ) {
  if ( values.size() == 0 ) {
    throw new std::string ( "Set no values" );
  }
  _d_elements = values;
}//set_values

std::vector< int >
SetDomain::get_values () const {
  return _d_elements;
}//get_values

size_t
SetDomain::get_size () const {
  return _d_elements.size ();
}//get_domain_size

bool
SetDomain::is_empty () const {
  return  ( _d_elements.size() == 0 );
}//is_empty

bool
SetDomain::is_singleton () const {
  return ( _d_elements.size() == 1 );
}//is_singleton

void
SetDomain::print () const {
  cout << "Set Domain: {";
  for ( auto x: _d_elements ) cout << x << " ";
  cout << "}\n";
  cout << "Size      : " << get_size() << "\n";
}//print

