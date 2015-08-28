//
//  greedy_neighborhood_evaluator.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/28/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "greedy_neighborhood_evaluator.h"
 
using namespace std;

GreedyNeighborhoodEvaluator:: GreedyNeighborhoodEvaluator () {
}

GreedyNeighborhoodEvaluator::~GreedyNeighborhoodEvaluator () {
}//~ GreedyNeighborhoodEvaluator
 
ObjectiveState 
GreedyNeighborhoodEvaluator::get_best_value ( std::vector< ObjectiveState >& obj_states )
{
	return get_best_value ( std::move ( obj_states ) );
}//get_best_value
 
ObjectiveState 
GreedyNeighborhoodEvaluator::get_best_value ( std::vector< ObjectiveState >&& obj_states )
{
	ObjectiveState best_obj_state;
	
	/*
	 * @note In this preliminary version we consider only the number
	 *       of unsatisfied constraints as objective value.
	 */
	std::size_t best_num_unsat = std::numeric_limits<int>::max();
	std::unordered_map < int, std::size_t > unsat_const_per_var;
	for ( auto& state : obj_states )
	{
		if ( state.number_unsat_constraint < best_num_unsat )
		{
			best_num_unsat = state.number_unsat_constraint;
			
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
	
	best_obj_state.obj_var_value = best_num_unsat;
	
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