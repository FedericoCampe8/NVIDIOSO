//
//  indomain_less_than_metric.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/26/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "indomain_less_than_metric.h"

InDomainLessThan::InDomainLessThan () {
  _dbg = "InDomainLessThan - ";
  _metric_type = ValueChoiceMetricType::INDOMAIN_MIN;
}//InDomainLessThan

int
InDomainLessThan::metric_value ( Variable * var, int comparator ) 
{
	int max_val = comparator - 1;
	int min_val = (var->domain_iterator)->min_val ();
	while ( max_val >= min_val && !((var->domain_iterator)->contains ( max_val )) )
	{
		--max_val;
	}
	
	// Reset max_val
	if ( max_val < min_val ) 
	{
		return comparator;
	}
	
	_current_val_lookup [ var->get_id() ] = max_val;
  	return max_val;
}//metric_value

int
InDomainLessThan::metric_value ( Variable * var ) 
{
	int max_val{};
	if ( _current_val_lookup.find ( var->get_id() ) == _current_val_lookup.end () )
	{
		max_val = (var->domain_iterator)->max_val ();
		_current_val_lookup [ var->get_id() ] = max_val;
		return max_val;
	}
	
	max_val = _current_val_lookup [ var->get_id() ] + 1;
	int min_val = (var->domain_iterator)->min_val ();
	while ( max_val >= min_val && !((var->domain_iterator)->contains ( max_val )) )
	{
		--max_val;
	}
	// Reset max_val
	if ( max_val < min_val ) 
	{
		max_val = (var->domain_iterator)->max_val ();
	}
	
	_current_val_lookup [ var->get_id() ] = max_val;
  	return max_val;
}//metric_value

void
InDomainLessThan::print () const {
  std::cout << "indomain_less_than" << std::endl;
}//print
