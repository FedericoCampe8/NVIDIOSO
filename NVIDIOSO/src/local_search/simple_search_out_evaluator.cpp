//
//  simple_search_out_evaluator.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/26/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "simple_search_out_evaluator.h"
#include "search_out_manager.h"

using namespace std;

SimpleSearchOutEvaluator::SimpleSearchOutEvaluator ( SearchOutManager * search_out_manager ) {	
	_id = glb_id_gen->get_int_id();	
	_search_out_manager = search_out_manager;
}//TimeSearchOutEvaluator

SimpleSearchOutEvaluator::~SimpleSearchOutEvaluator () {
}//~ TimeSearchOutEvaluator
 
void 
SimpleSearchOutEvaluator::notify ()
{
	_search_out_manager->notify_out ( _id );
}//notify 

void 
SimpleSearchOutEvaluator::set_limit_out ()
{
}//set_limit_out
 
void 
SimpleSearchOutEvaluator::upd_metric_value ()
{
}//upd_metric_value

std::size_t 
SimpleSearchOutEvaluator::get_id () const
{
	return _id;
}//get_id