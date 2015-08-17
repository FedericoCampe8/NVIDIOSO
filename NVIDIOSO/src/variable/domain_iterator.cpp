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
	/*
	 * @note very inefficient implementation.
	 * @todo change implementation.
	 */
	int min_val   = _domain->lower_bound();
	int max_val   = _domain->upper_bound();
	int num_elems;
	if ( min_val >= 0 && max_val >= 0 )
	{
		num_elems = max_val - min_val + 1;
	} 
	else if (  min_val < 0 && max_val >= 0 )
	{
		num_elems = max_val + abs ( min_val ) + 1;
	}
	else
	{// min_val < 0 && max_val < 0
		num_elems = abs ( min_val ) - abs ( max_val ) + 1;
	}
	
	int rand_val = min_val + (rand () % num_elems);
	if ( _domain->contains ( rand_val ) ) 
	{
		return rand_val;
	}
	else
	{
		int old_rand = rand_val;
		int half_size = (rand () % 100);
		if ( half_size >= 50 )
		{
			while ( ++rand_val <= max_val )
			{
				if ( _domain->contains ( rand_val ) ) 
				{
					return rand_val;
				}
			}
		}
		else
		{
			while ( --rand_val >= min_val )
			{
				if ( _domain->contains ( rand_val ) ) 
				{
					return rand_val;
				}
			}
		}
		
		// No value found on the right (left) or rand_val
		rand_val = old_rand;
		if ( half_size >= 50 )
		{
			while ( --rand_val >= min_val )
			{
				if ( _domain->contains ( rand_val ) ) 
				{
					return rand_val;
				}
			}
		}
		else
		{
			while ( ++rand_val <= max_val )
			{
				if ( _domain->contains ( rand_val ) ) 
				{
					return rand_val;
				}
			}
		}
	}
	
	std::string error_str = "DomainIterator::random_val - Error occurred";
	throw NvdException ( error_str.c_str() );
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