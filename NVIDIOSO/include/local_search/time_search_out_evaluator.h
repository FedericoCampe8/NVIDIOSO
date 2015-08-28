//
//  time_search_out_evaluator.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/26/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a search out evaluator evaluating time as out metric.
//  A search out evaluator implements a specific function associated to an "out"
//  parameter to set, update and query during the search process;
//

#ifndef __NVIDIOSO__time_search_out_evaluator__
#define __NVIDIOSO__time_search_out_evaluator__

#include "simple_search_out_evaluator.h"

class TimeSearchOutEvaluator : public SimpleSearchOutEvaluator {
private:
	double _abs_limit_value{};
	
	double get_seconds_from_time ( std::chrono::time_point<std::chrono::system_clock> p_time );
	
protected:

	//! Set limit depending on curent time plus _abs_limit_value 
	void set_limit_out () override;
	
	//! Update metric value depending on current time
	void upd_metric_value () override;
	
public:

	TimeSearchOutEvaluator ( SearchOutManager * search_out_manager );
	
	~TimeSearchOutEvaluator ();
	
	//! Reset state (metric) with current time value and new limit depending on it
	void reset_state () override;
	
	void print() const override;
};

#endif /* defined(__NVIDIOSO__time_search_out_evaluator__) */
