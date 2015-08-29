//
//  search_out_evaluator.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/22/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class declares the interface for a search out evaluator.
//  A search out evaluator implements a specific function associated to an "out"
//  parameter to set, update and query during the search process;
//

#ifndef __NVIDIOSO__search_out_evaluator__
#define __NVIDIOSO__search_out_evaluator__

#include "globals.h"

template<typename T> 
class SearchOutEvaluator {
protected:
	T _limit_value;
	T _metric_value; 
	
	//! Notify this evaluator: this is implementation dependent
	virtual void notify () = 0;
	
	//! Virtual function to set a limit depending on some internal state
	virtual void set_limit_out () = 0;
	
	//! Virtual function to update the metric value depending on some internal state
	virtual void upd_metric_value () = 0;
	 
public:
	virtual ~SearchOutEvaluator () {}
	
	//! Set a limit out value
	void set_limit_out ( T value )
	{
		_limit_value = value;
		set_limit_out ();
	}//set_limit_out
	
	//! Update a limit out value
	void upd_metric_value ( T value )
	{
		_metric_value = value;
		upd_metric_value();
		
		if ( _metric_value >= _limit_value )
		{
			notify ();
		}
	}//upd_limit_out
	
	T get_limit_out () const
	{
		return _limit_value;
	}//get_limit_out
	
	T& get_limit_out () 
	{
		return _limit_value;
	}//get_limit_out
	
	//! Check whether the limit is reached and, in case, notify
	virtual bool is_limit_reached () 
	{
		bool reached = (_metric_value >= _limit_value);
		if ( reached )
		{
			notify ();
		}
		return reached;
	}//is_limit_reached
	
	virtual std::size_t get_id () const = 0;
	
	virtual void reset_state () = 0;
	
	//! Print information about the search_out_manager
	virtual void print() const = 0;
};

#endif /* defined(__NVIDIOSO__search_out_manager__) */
