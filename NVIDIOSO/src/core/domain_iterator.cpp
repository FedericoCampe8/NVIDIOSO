//
//  domain_iterator.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "domain_iterator.h"

DomainIterator::DomainIterator ( IntDomainPtr domain ) {
  _domain = domain;
}//DomainIterator

DomainIterator::~DomainIterator () {
}//~DomainIterator

int
DomainIterator::min_val () const {
  return _domain->lower_bound();
}//min_val

int
DomainIterator::max_val () const {
  return _domain->upper_bound();
}//max_val

int
DomainIterator::random_val () const {
  throw std::string ( "Not yet implemented" );
}//random_val

