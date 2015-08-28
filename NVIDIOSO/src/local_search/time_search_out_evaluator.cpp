//
//  time_search_out_evaluator.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/26/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "time_search_out_evaluator.h"

using namespace std;
using namespace std::chrono;

TimeSearchOutEvaluator:: TimeSearchOutEvaluator ( SearchOutManager * search_out_manager ) :
	SimpleSearchOutEvaluator ( search_out_manager ) {	
}//TimeSearchOutEvaluator

TimeSearchOutEvaluator::~TimeSearchOutEvaluator () {
}//~TimeSearchOutEvaluator
 
double 
TimeSearchOutEvaluator::get_seconds_from_time ( std::chrono::time_point<std::chrono::system_clock> p_time )
{
	milliseconds ms = duration_cast<milliseconds>(p_time.time_since_epoch());
  	seconds s = duration_cast<seconds>(ms);
  	return s.count ();
}//get_seconds_from_time

void 
TimeSearchOutEvaluator::set_limit_out () 
{
	_abs_limit_value = _limit_value;
	
	std::chrono::time_point<std::chrono::system_clock> p_time;
    p_time = std::chrono::system_clock::now();
    milliseconds ms = duration_cast<milliseconds>(p_time.time_since_epoch());
  	seconds s = duration_cast<seconds>(ms);
	_limit_value += s.count(); 
}//reset_state
 
void 
TimeSearchOutEvaluator::upd_metric_value () 
{
	std::chrono::time_point<std::chrono::system_clock> p_time;
    p_time = std::chrono::system_clock::now();
    milliseconds ms = duration_cast<milliseconds>(p_time.time_since_epoch());
  	seconds s = duration_cast<seconds>(ms);
	_metric_value = s.count(); 
}//upd_metric_value

void 
TimeSearchOutEvaluator::reset_state () 
{
	std::chrono::time_point<std::chrono::system_clock> p_time;
    p_time = std::chrono::system_clock::now();
    
    milliseconds ms = duration_cast<milliseconds>(p_time.time_since_epoch());
  	seconds s = duration_cast<seconds>(ms);
    
	_metric_value = s.count(); 
	_limit_value  = _abs_limit_value + s.count();
}//reset_state

void 
TimeSearchOutEvaluator::print () const 
{
	cout << "TimeSearchOutEvaluator:\n";
	cout << "Absolute limit value:   \t" << _abs_limit_value << endl;
	cout << "Current out limit value:\t" << _limit_value     << endl;
	cout << "Current metric value:   \t" << _metric_value    << endl;
}//print