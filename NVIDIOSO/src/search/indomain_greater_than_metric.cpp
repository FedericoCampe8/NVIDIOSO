//
//  indomain_greater_than_metric.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/26/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "indomain_greater_than_metric.h"

InDomainGreaterThan::InDomainGreaterThan () {
  _dbg = "InDomainGreaterThan - ";
  _metric_type = ValueChoiceMetricType::INDOMAIN_MIN;
}//InDomainGreaterThan

int
InDomainGreaterThan::metric_value ( Variable * var, int comparator ) 
{
	int min_val = comparator + 1;
	int max_val = (var->domain_iterator)->max_val ();
	while ( min_val <= max_val && !((var->domain_iterator)->contains ( min_val )) )
	{
		++min_val;
	}
	// Reset min_val
	if ( min_val > max_val ) 
	{
		return comparator;
	}
	
	_current_val_lookup [ var->get_id() ] = min_val;
  	return min_val;
}//metric_value

int
InDomainGreaterThan::metric_value ( Variable * var ) 
{
	int min_val{};
	if ( _current_val_lookup.find ( var->get_id() ) == _current_val_lookup.end () )
	{
		min_val = (var->domain_iterator)->min_val ();
		_current_val_lookup [ var->get_id() ] = min_val;
		return min_val;
	}
	
	min_val = _current_val_lookup [ var->get_id() ] + 1;
	int max_val = (var->domain_iterator)->max_val ();
	while ( min_val <= max_val && !((var->domain_iterator)->contains ( min_val )) )
	{
		++min_val;
	}
	// Reset min_val
	if ( min_val > max_val ) 
	{
		min_val = (var->domain_iterator)->min_val ();
	}
	
	_current_val_lookup [ var->get_id() ] = min_val;
  	return min_val;
}//metric_value

void
InDomainGreaterThan::print () const {
  std::cout << "indomain_greater_than" << std::endl;
}//print
