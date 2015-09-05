//
//  greedy_neighborhood_evaluator.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/28/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "greedy_neighborhood_evaluator.h"
 
using namespace std;

GreedyNeighborhoodEvaluator::GreedyNeighborhoodEvaluator () :
	_objective_type ( ObjectiveValueType::SAT_CON ),
	_minimize       ( true ) {
}

GreedyNeighborhoodEvaluator::~GreedyNeighborhoodEvaluator () {
}//~ GreedyNeighborhoodEvaluator
 
void 
GreedyNeighborhoodEvaluator::set_objective ( ObjectiveValueType ovt )
{
	_objective_type = ovt;
}//set_objective

void 
GreedyNeighborhoodEvaluator::set_minimize_objective ()
{
	_minimize = true;
}//set_minimize_objective

void 
GreedyNeighborhoodEvaluator::set_maximize_objective ()
{
	_minimize = false;
}//set_minimize_objective

ObjectiveState 
GreedyNeighborhoodEvaluator::get_best_value ( std::vector< ObjectiveState >& obj_states )
{
	return get_best_value ( std::move ( obj_states ) );
}//get_best_value
 
ObjectiveState 
GreedyNeighborhoodEvaluator::get_best_value ( std::vector< ObjectiveState >&& obj_states )
{
	// Best state to return according to the evaluator function (greedy)
	ObjectiveState best_obj_state;
	
	int best_value, current_val;
	if ( _minimize )
	{
		best_value = std::numeric_limits<int>::max();
	}
	else
	{
		best_value = std::numeric_limits<int>::min();
	}
	
	bool upd_value;
	std::unordered_map < int, std::size_t > unsat_const_per_var;
	for ( auto& state : obj_states )
	{
		upd_value  = false;
		if ( _objective_type == ObjectiveValueType::SAT_CON )
		{
			current_val = state.number_unsat_constraint;
		}
		else if ( _objective_type == ObjectiveValueType::SAT_VAL )
		{
			current_val = (int) (state.unsat_value * 10000) / 10;
		}
		else if ( _objective_type == ObjectiveValueType::OBJ_VAR )
		{
			current_val = state.obj_var_value;
		}
		else
		{
			current_val = state.number_unsat_constraint;
		}
		
		if ( ( _minimize  && current_val <= best_value ) ||
			 ( !_minimize && current_val >= best_value ) )
		{
			best_value = current_val;
			upd_value  = true;
		}
		
		if ( upd_value )
		{
			// Sanity check
			assert ( state.neighborhood_index.size () == state.neighborhood_values.size () );
			
			// Update best values for the variables in the neighborhood
			for ( int i = 0; i < state.neighborhood_index.size (); ++i )
			{
				unsat_const_per_var [ state.neighborhood_index [ i ] ] = state.neighborhood_values [ i ];
			}
		}
	}
	
	std::vector< int > neighborhood_idx;
	std::vector< int > neighborhood_val;
	
	for ( auto& val : unsat_const_per_var )
	{
		neighborhood_idx.push_back ( val.first );
		neighborhood_val.push_back ( val.second );
	}
	
	best_obj_state.obj_var_value = best_value;
	
	// Set the neighborhood corresponding to the given state 
	best_obj_state.neighborhood_index  = neighborhood_idx;
	
	// Set the labels corresponding to the given state
	best_obj_state.neighborhood_values = neighborhood_val;
	
	// Set time stamp
	best_obj_state.timestamp = std::time ( 0 ); 
	
	return best_obj_state;

}//get_best_value

void 
GreedyNeighborhoodEvaluator::reset_evaluator ()
{
}//reset_evaluator

void 
GreedyNeighborhoodEvaluator::print () const 
{
	cout << "GreedyNeighborhoodEvaluator\n";
}//print