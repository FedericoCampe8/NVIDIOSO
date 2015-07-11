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
  if ( lower_bound () == upper_bound () ) 
  {
  	assert( get_size() == 1 );
  }
  
  return ( lower_bound() == upper_bound() );
}//is_singleton

//! Returns true if this is a numeric finite domain
bool
IntDomain::is_numeric () const {
  return true;
}//is_numeric

void 
IntDomain::set_domain_status ( void * domain )
{
}//set_domain_status

size_t
IntDomain::get_domain_size () const {
  return 0;
}//get_domain_size

const void *
IntDomain::get_domain_status () const {
  return nullptr;
}//get_domain_status

string
IntDomain::get_string_representation () const {
  string domain_str = "";
  for ( int i = lower_bound(); i <= upper_bound(); i++ ) {
    if ( contains ( i ) ) {
      ostringstream convert;
      convert << i;
      domain_str += convert.str();
      domain_str += " ";
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




