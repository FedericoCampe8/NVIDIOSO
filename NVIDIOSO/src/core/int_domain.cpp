//
//  int_domain.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 09/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "int_domain.h"

using namespace std;

IntDomain::IntDomain () {
  _dbg = "IntDomain - ";
  _dom_type = DomainType::INTEGER;
  
}//IntDomain

IntDomain::~IntDomain () {
}//~IntDomain

bool
IntDomain::is_empty () const {
  return ( get_size() == 0 );
}//is_empty

bool
IntDomain::is_singleton () const {
  
  // Consistency check
  if ( lower_bound() == upper_bound() ) {
    assert( get_size() == 1 );
  }
  
  return ( lower_bound() == upper_bound() );
}//is_singleton

//! Returns true if this is a numeric finite domain
bool
IntDomain::is_numeric () const {
  return true;
}//is_numeric

string
IntDomain::get_string_representation () const {
  string domain_str = "";
  for ( int i = lower_bound(); i <= upper_bound(); i++ ) {
    if ( contains ( i ) ) {
      ostringstream convert;
      convert << i;
      domain_str += convert.str();
    }
  }
  return domain_str;
}//get_string_representation

void
IntDomain::print () const {
  cout << "Int Domain: ";
  cout << lower_bound() << ".." << upper_bound() << "\n";
  cout << "Size      : " << get_size() << "\n";
}//print




