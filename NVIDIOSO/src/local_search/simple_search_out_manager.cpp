//
//  simple_search_out_manager.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/25/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "simple_search_out_manager.h"
#include "time_search_out_evaluator.h"
#include "counter_search_out_evaluator.h"

using namespace std;

SimpleSearchOutManager::SimpleSearchOutManager () :
	_dbg 		( "SimpleSearchOutManager - "),
	_search_out ( false ) {
	initialize_manager ();
}//SimpleSearchOutManager

SimpleSearchOutManager::~SimpleSearchOutManager () {
}//~SimpleSearchOutManager

void
SimpleSearchOutManager::initialize_manager ()
{ 
	add_out_evaluator ( new TimeSearchOutEvaluator    ( this ), "time_out" ); 
	add_out_evaluator ( new CounterSearchOutEvaluator ( this ), "num_iterative_improving_out" );
	add_out_evaluator ( new CounterSearchOutEvaluator ( this ), "num_solution_out" );
	add_out_evaluator ( new CounterSearchOutEvaluator ( this ), "num_restarts_out" );
	add_out_evaluator ( new CounterSearchOutEvaluator ( this ), "num_nodes_out" );
	add_out_evaluator ( new CounterSearchOutEvaluator ( this ), "num_wrong_decisions_out" );
}//initialize_manager
   
void  
SimpleSearchOutManager::notify_out ( std::size_t eval_id )
{
	auto it = _out_evaluators.find ( eval_id );
	if ( it != _out_evaluators.end() )
	{
		if ( _out_evaluators [ eval_id ].first )
		{
			LogMsg << _dbg + "search out notification received." << endl;
			_search_out = true;
		}
	}	
}//notify_out
 
void 
SimpleSearchOutManager::force_out ()
{
	_search_out = true;
}//force_out

bool 
SimpleSearchOutManager::search_out ()
{
	bool set_out = _search_out;
	if ( !_search_out )
	{
		for ( auto& evals : _out_evaluators )
		{
			if ( (evals.second.second)->is_limit_reached () )
			{
				set_out = true;
				break;
			}
		}
	}
	 
	_search_out = false;
	return set_out; 
}//search_out 
 
void 
SimpleSearchOutManager::reset_out_evaluator ( std::size_t eval_id )
{
	auto it = _out_evaluators.find ( eval_id );
	if ( it != _out_evaluators.end() )
	{
		(_out_evaluators [ eval_id ].second)->reset_state ();
	}
}//reset_out_evaluator

void 
SimpleSearchOutManager::reset_out_evaluator ()
{
	for ( auto& eval : _out_evaluators )
	{
		(eval.second.second)->reset_state ();
	}
}//reset_out_evaluator

void 
SimpleSearchOutManager::add_out_evaluator ( SimpleSearchOutEvaluator * out_eval, std::string eval_str_id )
{
	_string_eval_lookup [ eval_str_id ] = out_eval->get_id ();
	SearchOutManager::add_out_evaluator ( out_eval );
}//add_out_evaluator

bool 
SimpleSearchOutManager::is_active_evaluator ( std::size_t eval_id ) const
{
	auto it = _out_evaluators.find ( eval_id );
	if ( it != _out_evaluators.end() )
	{
		return _out_evaluators.at ( eval_id ).first;
	}
	return false;
}//is_active_evaluator
 
void 
SimpleSearchOutManager::activate_out_evaluator ( std::size_t eval_id )
{
	auto it = _out_evaluators.find ( eval_id );
	if ( it != _out_evaluators.end() )
	{
		_out_evaluators [ eval_id ].first = true;
	}
}//activate_out_evaluator

void 
SimpleSearchOutManager::deactivate_out_evaluator ( std::size_t eval_id )
{ 
	auto it = _out_evaluators.find ( eval_id );
	if ( it != _out_evaluators.end() )
	{
		_out_evaluators [ eval_id ].first = false;
	}
}//activate_out_evaluator

void 
SimpleSearchOutManager::set_num_restarts_out ( std::size_t num_sol )
{
	if ( num_sol == 0 )
	{// Deactivate evaluator
		deactivate_out_evaluator ( _string_eval_lookup [ "num_restarts_out" ] );
	}
	else
	{
		set_out_value ( _string_eval_lookup [ "num_restarts_out" ], num_sol );
		activate_out_evaluator ( _string_eval_lookup [ "num_restarts_out" ] );
	}
}//set_num_restarts_out

void 
SimpleSearchOutManager::set_num_iterative_improvings_out ( std::size_t ii )
{ 
	if  ( ii == 0 )
	{// Deactivate evaluator
		deactivate_out_evaluator ( _string_eval_lookup [ "num_iterative_improving_out" ] );
	}
	else
	{
		set_out_value ( _string_eval_lookup [ "num_iterative_improving_out" ], ii );
		activate_out_evaluator ( _string_eval_lookup [ "num_iterative_improving_out" ] );
	}
}//set_num_iterative_improving_out

void 
SimpleSearchOutManager::set_num_solutions_out ( std::size_t num_sol )
{
	if ( num_sol == 0 )
	{// Deactivate the evaluator
		deactivate_out_evaluator ( _string_eval_lookup [ "num_solution_out" ] );
	}
	else
	{
		set_out_value ( _string_eval_lookup [ "num_solution_out" ], num_sol );
		activate_out_evaluator ( _string_eval_lookup [ "num_solution_out" ] );
	}
}//set_num_solution_out

void 
SimpleSearchOutManager::set_time_out ( double timeout )
{
	if ( timeout < 0.0 )
	{// Deactivate the evaluator
		deactivate_out_evaluator ( _string_eval_lookup [ "time_out" ] );
	}
	else
	{
		set_out_value ( _string_eval_lookup [ "time_out" ], timeout );
		activate_out_evaluator ( _string_eval_lookup [ "time_out" ] );
	}
}//set_time_out
 
void  
SimpleSearchOutManager::set_num_nodes_out ( std::size_t out_n )
{
	if ( out_n == 0 )
	{// Deactivate the evaluator
		deactivate_out_evaluator ( _string_eval_lookup [ "num_nodes_out" ] );
	}
	else
	{
		set_out_value ( _string_eval_lookup [ "num_nodes_out" ], out_n );
		activate_out_evaluator ( _string_eval_lookup [ "num_nodes_out" ] );
	}
}//set_num_nodes_out

void 
SimpleSearchOutManager::set_num_wrong_decisions_out ( std::size_t out_w )
{
	if ( out_w == 0 )
	{// Deactivate the evaluator
		deactivate_out_evaluator ( _string_eval_lookup [ "num_wrong_decisions" ] );
	}
	else
	{
		set_out_value ( _string_eval_lookup [ "num_wrong_decisions" ], out_w );
		activate_out_evaluator ( _string_eval_lookup [ "num_wrong_decisions" ] );
	}
}//set_num_wrong_decisions_out

void 
SimpleSearchOutManager::upd_restarts ( std::size_t num_sol )
{
	if ( is_active_evaluator ( _string_eval_lookup [ "num_restarts_out" ] ) )
	{
		upd_metric_value ( _string_eval_lookup [ "num_restarts_out" ], num_sol );
	}
}//set_num_restarts_out

void 
SimpleSearchOutManager::upd_iterative_improvings_steps ( std::size_t ii )
{
	if ( is_active_evaluator ( _string_eval_lookup [ "num_iterative_improving_out" ] ) )
	{
		upd_metric_value ( _string_eval_lookup [ "num_iterative_improving_out" ], ii ); 
	}
}//set_num_iterative_improving_out

void 
SimpleSearchOutManager::upd_solutions ( std::size_t num_sol )
{
	if ( is_active_evaluator ( _string_eval_lookup [ "num_solution_out" ] ) )
	{
		upd_metric_value ( _string_eval_lookup [ "num_solution_out" ], num_sol );
	}
}//set_num_solution_out

void 
SimpleSearchOutManager::upd_time ( double timeout )
{
	if ( is_active_evaluator ( _string_eval_lookup [ "time_out" ] ) )
	{
		upd_metric_value ( _string_eval_lookup [ "time_out" ], timeout );
	}
}//set_time_out

void 
SimpleSearchOutManager::upd_nodes ( std::size_t out_n )
{
	if ( is_active_evaluator ( _string_eval_lookup [ "num_nodes_out" ] ) )
	{
		upd_metric_value ( _string_eval_lookup [ "num_nodes_out" ], out_n );
	}
}//set_restarts_out

void 
SimpleSearchOutManager::upd_wrong_decisions ( std::size_t out_w )
{
	if ( is_active_evaluator ( _string_eval_lookup [ "num_wrong_decisions" ] ) )
	{
		upd_metric_value ( _string_eval_lookup [ "num_wrong_decisions" ], out_w );
	}
}//set_num_wrong_decisions_out

void 
SimpleSearchOutManager::print() const
{
}//print

