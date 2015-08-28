//
//  counter_search_out_evaluator.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/26/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a search out evaluator evaluating a counter as out metric.
//  A search out evaluator implements a specific function associated to an "out"
//  parameter to set, update and query during the search process;
//

#ifndef __NVIDIOSO__counter_search_out_evaluator__
#define __NVIDIOSO__counter_search_out_evaluator__

#include "simple_search_out_evaluator.h"

class CounterSearchOutEvaluator : public SimpleSearchOutEvaluator {
public:

	CounterSearchOutEvaluator ( SearchOutManager * search_out_manager );
	
	~CounterSearchOutEvaluator ();
	
	//! Reset state (metric) with current time value and new limit depending on it
	void reset_state () override;
	
	void print() const override;
};

#endif /* defined(__NVIDIOSO__counter_search_out_evaluator__) */
