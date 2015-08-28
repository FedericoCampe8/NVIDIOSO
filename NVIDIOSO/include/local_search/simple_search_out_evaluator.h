//
//  simple_search_out_evaluator.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/22/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a base abstract class for a search out evaluator 
//  based on double value types.
//  A search out evaluator implements a specific function associated to an "out"
//  parameter to set, update and query during the search process;
//

#ifndef __NVIDIOSO__simple_search_out_evaluator__
#define __NVIDIOSO__simple_search_out_evaluator__

#include "search_out_evaluator.h"

class SearchOutManager;

class SimpleSearchOutEvaluator : public SearchOutEvaluator<double> {
protected:
	
	std::size_t _id;
	
	SearchOutManager * _search_out_manager;
	
	/**
	 * Notify search_out_manager.
	 * @note This is invoked when an out limit is reached, 
	 *       search_out_manager is notified about out.
	 */
	void notify () override;
	
	//! Set limit depending on internal state
	void set_limit_out () override;
	
	//! Update metric value depending on internal state
	void upd_metric_value () override;
	
	SimpleSearchOutEvaluator ( SearchOutManager * search_out_manager );
 	
public:

	using SearchOutEvaluator::set_limit_out;
	using SearchOutEvaluator::upd_metric_value;
	
	~SimpleSearchOutEvaluator ();
	
	//! Get unique id of this evaluator
	std::size_t get_id () const override;
};

#endif /* defined(__NVIDIOSO__simple_search_out_evaluator__) */
