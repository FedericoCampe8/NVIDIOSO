//
//  domain_iterator.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/08/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "domain_iterator.h"

DomainIterator::DomainIterator ( IntDomainPtr domain ) {
  _domain = domain;
}//DomainIterator

DomainIterator::~DomainIterator () {
}//~DomainIterator

bool
DomainIterator::is_numeric () const 
{
  return _domain->is_numeric();
}//is_numeric

int
DomainIterator::min_val () const 
{
	return _domain->lower_bound();
}//min_val

int
DomainIterator::max_val () const 
{
	return _domain->upper_bound();
}//max_val

int
DomainIterator::random_val () const 
{
	throw std::string ( "DomainIterator::random_val - Not yet implemented" );
}//random_val

size_t
DomainIterator::domain_size () const 
{
	return _domain->get_size();
}//domain_size

std::pair<size_t, const void *>
DomainIterator::get_domain_status () const {
  return std::make_pair ( _domain->get_domain_size () , _domain->get_domain_status () );
}//get_domain_status

std::string
DomainIterator::get_string_representation () const {
  return _domain->get_string_representation ();
}//get_string_representation

void 
DomainIterator::set_domain_status ( void * concrete_domain ) 
{
	_domain->set_domain_status ( concrete_domain );
}//set_domain_status