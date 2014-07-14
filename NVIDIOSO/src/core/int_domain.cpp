//
//  int_domain.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 09/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "int_domain.h"

using namespace std;

IntDomain::IntDomain () :
_lower_bound (  0 ),
_upper_bound ( -1 ) {
  _dbg = "IntDomain - ";
  _dom_type = DomainType::INTEGER;
}//IntDomain

IntDomain::~IntDomain () {
}//~IntDomain

int
IntDomain::get_lower_bound () const {
  return _lower_bound;
}//get_lower_bound

int
IntDomain::get_upper_bound () const {
  return _upper_bound;
}//get_upper_bound

bool
IntDomain::is_empty () const {
  return ( get_size() == 0 );
}//is_empty

bool
IntDomain::is_singleton () const {
  // Consistency check
  if ( _lower_bound == _upper_bound ) {
    assert( get_size() == 1 );
  }
  return ( _lower_bound == _upper_bound );
}//is_singleton

void
IntDomain::print () const {
  cout << "Int Domain: ";
  cout << _lower_bound << ".." << _upper_bound << "\n";
  cout << "Size      : " << get_size() << "\n";
}//print




