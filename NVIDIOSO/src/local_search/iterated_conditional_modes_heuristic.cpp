//
//  iterated_conditional_modes_heuristic.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/24/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "iterated_conditional_modes_heuristic.h"
#include "greedy_neighborhood_evaluator.h" 
#include "metric_inc.h"

using namespace std;

IteratedConditionalModesHeuristic::IteratedConditionalModesHeuristic ( std::vector< Variable* > vars,
																	   Variable * obj_var,
																	   bool minimize,
																	   bool use_constraint_satisfiability_value, 
																	   int percentage_on_sat,
																	   VariableChoiceMetric * var_cm, 
																	   ValueChoiceMetric    * val_cm ) :
	SimpleLocalSearchHeuristic ( vars, obj_var ) {
	_dbg = "IteratedConditionalModesHeuristic - "; 
	 
	// Set status of the variables
	for ( int i = 0; i < vars.size(); i++ )
		_sampling_variable_status.push_back ( -1 ); 
	
	// Set starting neighborhood
	_neighborhood = starting_neighborhood ();
	
	// Set heuristic for exploring neighborhoods
	if ( var_cm != nullptr || val_cm != nullptr )
	{
		LogMsg << _dbg  << "WARNING: setting metrics different from default" << endl;
	}
	else
	{
		val_cm = new InDomainGreaterThan ();
	}	
	set_neighborhood_heuristic ( var_cm, val_cm );
	
	// Set evaluator function
	set_neighborhood_evaluator ( minimize, use_constraint_satisfiability_value, percentage_on_sat ) ;
}

IteratedConditionalModesHeuristic::~IteratedConditionalModesHeuristic () {
}//~IteratedConditionalModesHeuristic

void 
IteratedConditionalModesHeuristic::reset_state ()
{
	SimpleLocalSearchHeuristic::reset_state ();
	
	// Set status of the variables
	_sampling_variable_status.clear();
	for ( int i = 0; i < _fd_variables.size(); i++ )
		_sampling_variable_status.push_back ( -1 ); 
}//reset_state

void 
IteratedConditionalModesHeuristic::set_neighborhood_heuristic ( VariableChoiceMetric * var_cm, ValueChoiceMetric * val_cm )
{
	_neighborhood_heuristic = 
	std::move ( std::unique_ptr<NeighborhoodHeuristic> ( new NeighborhoodHeuristic ( _fd_variables, var_cm, val_cm ) ) );
}//set_neighborhood_heuristic

void 
IteratedConditionalModesHeuristic::set_neighborhood_evaluator ( bool minimize, bool use_constraint_satisfiability_value, int percentage_on_sat )
{
	_neighborhood_evaluator = 
	std::move ( std::unique_ptr<GreedyNeighborhoodEvaluator> ( new GreedyNeighborhoodEvaluator () ) ); 
	
	if ( _obj_variable != nullptr )
	{
		_neighborhood_evaluator->set_objective ( ObjectiveValueType::OBJ_VAR );
	}
	else 
	{
		if ( use_constraint_satisfiability_value )
		{
			_neighborhood_evaluator->set_objective ( ObjectiveValueType::SAT_VAL );
		}
		else
		{
			_neighborhood_evaluator->set_objective ( ObjectiveValueType::SAT_CON );
		}
	}
	
	if ( minimize )
	{
		_neighborhood_evaluator->set_minimize_objective ();
	}
	else
	{
		_neighborhood_evaluator->set_maximize_objective ();
	}
}//set_neighborhood_evaluator
 
void  
IteratedConditionalModesHeuristic::notify_on_var_assignment ( int var_index ) 
{
	// Sanity check
  	assert ( var_index >= 0 && var_index < _sampling_variable_status.size() );
  	
  	// First time it is called, set it
  	if ( _sampling_variable_status [ var_index ] == -1 )
  	{
  		_sampling_variable_status [ var_index ] = _fd_variables [ var_index ]->get_size ();
  	}
	_sampling_variable_status [ var_index ]--;
}//neighborhood_assignment_on_var

bool 
IteratedConditionalModesHeuristic::neighborhood_last_assignment_on_var ( int var_index )
{
	// Sanity check
  	assert ( var_index >= 0 && var_index < _sampling_variable_status.size() );
	return _sampling_variable_status [ var_index ] == 1;
}

bool 
IteratedConditionalModesHeuristic::neighborhood_complete_assignment_on_var ( int var_index )
{
	// Sanity check
  	assert ( var_index >= 0 && var_index < _sampling_variable_status.size() );
	return _sampling_variable_status [ var_index ] == 0;
}

std::vector<int> 
IteratedConditionalModesHeuristic::starting_neighborhood ()
{
	std::vector<int> starting_neighborhood ( 1, 0 );
	return starting_neighborhood;
}

void
IteratedConditionalModesHeuristic::print () const
{
	// Sanity check
	assert ( _neighborhood_heuristic != nullptr && _neighborhood_evaluator != nullptr );
	cout << "IteratedConditionalModesHeuristic\n";
	_neighborhood_heuristic->print ();
	_neighborhood_evaluator->print();
}//print