//
//  counter_search_out_evaluator.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/25/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "counter_search_out_evaluator.h"

using namespace std;

CounterSearchOutEvaluator:: CounterSearchOutEvaluator ( SearchOutManager * search_out_manager ) :
	SimpleSearchOutEvaluator ( search_out_manager ) {	
	// Default: no limit
	_metric_value = -1;
	_limit_value  = 0;
}//CounterSearchOutEvaluator

CounterSearchOutEvaluator::~CounterSearchOutEvaluator () {
}//~CounterSearchOutEvaluator
 
void 
CounterSearchOutEvaluator::reset_state () 
{
	if ( _limit_value >= 0 )
	{
		_metric_value = 0; 
	}
}//reset_state

void 
CounterSearchOutEvaluator::print () const 
{
	cout << "CounterSearchOutEvaluator:\n";
	cout << "Current out limit value:\t" << _limit_value     << endl;
	cout << "Current metric value:   \t" << _metric_value    << endl;
}//print