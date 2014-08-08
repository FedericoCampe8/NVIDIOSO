//
//  bool_domain.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 10/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "bool_domain.h"

using namespace std;

BoolDomain::BoolDomain () :
_bool_value ( BoolValue::UNDEF_VALUE ) {
}//BoolDomain

BoolDomain::~BoolDomain () {
}//~BoolDomain

DomainPtr
BoolDomain::clone () const {
  return clone_impl ();
}//clone

DomainPtr
BoolDomain::clone_impl () const {
  return ( shared_ptr<BoolDomain> ( new BoolDomain ( *this ) ) );
}//clone_impl

EventType
BoolDomain::get_event () const {
  return  EventType::OTHER_EVT;
}//get_event

size_t
BoolDomain::get_size () const {
  if ( is_singleton() ) return 1;
  else                  return 2;
}//get_size

bool
BoolDomain::is_singleton () const {
  return ( _bool_value != BoolValue::UNDEF_VALUE );
}//is_singleton

// Returns true if the domain is empty
bool
BoolDomain::is_empty () const {
  return ( _bool_value == BoolValue::EMPTY_VALUE );
}//is_empty

bool
BoolDomain::is_numeric () const {
  return false;
}//is_numeric

std::string
BoolDomain::get_string_representation () const {
  throw string ( "Not yet implemented" );
}//get_string_representation

void
BoolDomain::print () const {
  cout << "Bool Domain: ";
  switch ( _bool_value ) {
    case BoolValue::TRUE_VALUE:
      cout << "True\n";
      break;
    case BoolValue::FALSE_VALUE:
      cout << "False\n";
      break;
    case BoolValue::EMPTY_VALUE:
      cout << "Empty\n";
      break;
    default:
      cout << "Undef\n";
      break;
  }
}//print

