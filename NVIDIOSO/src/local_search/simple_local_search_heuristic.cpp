//
//  simple_local_search_heuristic.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/23/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "simple_local_search_heuristic.h"
#include "simple_search_memory_manager.h"

using namespace std;

SimpleLocalSearchHeuristic::SimpleLocalSearchHeuristic ( std::vector< Variable* > vars, Variable * obj_var ) :
	_current_index ( 0 ) {
	
	// Set objective variable
	_obj_variable = obj_var;
	
	// Set variables 
	_fd_variables = vars;
	
	// Instantiate e new memory manager
	_memory_manager = std::move ( std::unique_ptr<SimpleSearchMemoryManager> ( new SimpleSearchMemoryManager () ) );
	
	//Init _explored_variables mapping
	for ( int i = 0; i < vars.size(); i++ )
	{
		_explored_variables [ i ] = false;
	}
}

SimpleLocalSearchHeuristic::~SimpleLocalSearchHeuristic () {
}//~SimpleLocalSearchHeuristic

void
SimpleLocalSearchHeuristic::reset_state ()
{ 
	// Sanity check 
	assert ( _neighborhood_heuristic != nullptr );
	assert ( _memory_manager != nullptr );
	
	_current_index = 0;
	_neighborhood_idx.clear ();
	_neighborhood_val.clear ();
	_neighborhood_var.clear ();
	_neighborhood  = starting_neighborhood ();
	
	_neighborhood_heuristic->set_index ( 0 );
	_neighborhood_evaluator->reset_evaluator ();
	_memory_manager->clear_memory ();
	
	reset_explored_variables ();
}//reset_state

void
SimpleLocalSearchHeuristic::reset_explored_variables () 
{
	for ( auto& val : _explored_variables )
	{
		val.second = false;
	}
}//reset_explored_variables

void 
SimpleLocalSearchHeuristic::set_search_variables ( std::vector< Variable* >& vars, Variable * obj_var )
{
	// Sanity check 
	assert ( _neighborhood_heuristic != nullptr );
	
	// Reset the current state
	reset_state ();
	
	// Set obj var if not NULL
	if ( obj_var != nullptr )
		_obj_variable = obj_var;
		
	// Set new set of variables to explore
	_fd_variables = vars;
	_neighborhood_heuristic->set_search_variables ( _fd_variables );
	
	// Reset _explored_variables mapping
	for ( int i = 0; i < vars.size(); i++ )
	{
		_explored_variables [ i ] = false;
	}
}//set_search_variables
  
void 
SimpleLocalSearchHeuristic::update_objective ( std::size_t num_unsat, double unsat_level )
{
	ObjectiveState obj_state;
	if ( _obj_variable != nullptr )
	{ 
		obj_state.obj_var_value = _obj_variable->domain_iterator->min_val ();
	} 
	
	obj_state.number_unsat_constraint = num_unsat;
	obj_state.unsat_value             = unsat_level;
	
	// Set the neighborhood corresponding to the given state 
	obj_state.neighborhood_index  = _neighborhood_idx;
	
	// Set the labels corresponding to the given state
	obj_state.neighborhood_values = _neighborhood_val;
	
	// Set time stamp
	obj_state.timestamp = std::time ( 0 ); 
	 
	// Record the new state 
	_memory_manager->record_state ( obj_state );
}//update_objective
   
bool 
SimpleLocalSearchHeuristic::is_explored_var ( int idx ) const
{
	auto it = _explored_variables.find ( idx );
	if ( it != _explored_variables.end() )
	{
		return _explored_variables.at ( idx );
	}
	std::string err_msg{"SimpleLocalSearchHeuristic::is_explored_var idx not found"};
	throw NvdException ( err_msg.c_str() ); 
}

void
SimpleLocalSearchHeuristic::set_explored_var ( int idx, bool val )
{
	auto it = _explored_variables.find ( idx );
	if ( it != _explored_variables.end() )
	{
		_explored_variables [ idx ] = val;
	}
}

int 
SimpleLocalSearchHeuristic::get_index () const
{

	if ( _current_index == _neighborhood.size() )
	{
		_current_index = 0;
	}
	
	while (  _current_index < _neighborhood.size() && 
	 		is_explored_var (_neighborhood [ _current_index ]) )
	{
		_current_index++;
	}
	
	if ( _current_index == _neighborhood.size() )
	{
		return -1;
	}
	
	int idx_to_return = _current_index;
	
	// Prepare _current_index for next iteration
	++_current_index;
	
	return _current_index;
}//get_index

Variable * 
SimpleLocalSearchHeuristic::get_choice_variable ( int index ) 
{
	if ( _current_index < 0 || _current_index >= _fd_variables.size() )
	{
		return nullptr;
	}
	return _fd_variables [ _current_index ];
}//get_choice_variable

int 
SimpleLocalSearchHeuristic::get_choice_value ()
{
	// Sanity checks 
	assert ( _neighborhood_heuristic != nullptr );
	assert ( _neighborhood_evaluator != nullptr );
	
	// Sanity check
  	if ( (_current_index < 0) || (_current_index >= _fd_variables.size()) ) 
  	{
		std::string err_mes{"SimpleLocalSearchHeuristic::get_choice_value - index not consistent"};
    	throw  NvdException ( err_mes.c_str() );
  	}
  
  	int value;
  	int index_to_add = -1;
  	int first_index  = _neighborhood_heuristic->get_index ();
	if ( !is_explored_var ( _current_index ) )
	{
		/*
		 * If we have a fully explored var for var idx according to 
		 * the current local search strategy, set var idx as explored.
		 */
		if ( neighborhood_complete_assignment_on_var ( _current_index ) )
		{
			value = _neighborhood_evaluator->get_best_value ( 
					_memory_manager-> get_state ( _current_index ) ).neighborhood_values[0];
			
			// Remove idx from _neighborhood
			set_explored_var ( _current_index );
		}
		else
		{	
			_neighborhood_heuristic->set_index ( _current_index );
				
			// To make it faster on the last value for a variable, prepare next one
			if ( neighborhood_last_assignment_on_var ( _current_index ) ) 
			{
				index_to_add = _neighborhood_heuristic->get_next_index ();
			}
			value = _neighborhood_heuristic->get_choice_value ();
		}
	} 
	
	// Reset index in neighborhood_heuristic
	_neighborhood_heuristic->set_index ( first_index );
	
	// Extend neighborhood according to the local search strategy 
	if ( index_to_add >= 0 )
	{
		_neighborhood.push_back ( index_to_add );
	}
	
	return value;
}//get_choice_value

std::vector<int> 
SimpleLocalSearchHeuristic::ls_get_index () const
{
	_neighborhood_idx.clear ();
	for ( auto& idx : _neighborhood )
	{
		if ( !is_explored_var ( idx ) )
		{
			_neighborhood_idx.push_back ( idx );
		} 
	}
	return _neighborhood_idx;
}//ls_get_index
 
std::vector<Variable *> 
SimpleLocalSearchHeuristic::ls_get_choice_variable ( std::vector< int > index )
{
	_neighborhood_var.clear();
	for ( auto& current_index : index )
	{
		// Consistency check
  		if ( (current_index < 0) || (current_index >= _fd_variables.size()) ) 
  		{
    		throw  NvdException ( (_dbg + " ls_get_choice_variable: index not consistent").c_str() );
  		}
  		_neighborhood_var.push_back (  _fd_variables[ current_index ] );
	}
	return _neighborhood_var;
}//ls_get_choice_variable 

std::vector<int>
SimpleLocalSearchHeuristic::ls_get_choice_value ()
{
	// Sanity checks 
	assert ( _neighborhood_heuristic != nullptr );
	assert ( _neighborhood_evaluator != nullptr );
	
	int first_index = _neighborhood_heuristic->get_index ();
	_neighborhood_val.clear();
	
	std::unordered_set<int> index_to_add;
	for ( auto& idx : _neighborhood )
	{
		if ( !is_explored_var ( idx ) )
		{ 
			/*
			 * If we have a fully explored var for var idx according to 
			 * the current local search strategy, set var idx as explored.
			 */
			if ( neighborhood_complete_assignment_on_var ( idx ) )
			{
				_neighborhood_val.push_back ( 
				_neighborhood_evaluator->get_best_value ( _memory_manager->get_state ( idx ) ).neighborhood_values[0] ); 
				
				// Remove idx from _neighborhood
				set_explored_var ( idx );
			}
			else
			{	
				_neighborhood_heuristic->set_index ( idx );
				
				// To make it faster on the last value for a variable, prepare next one
				if ( neighborhood_last_assignment_on_var ( idx ) ) 
				{
					index_to_add.insert ( _neighborhood_heuristic->get_next_index () );
				}
				_neighborhood_val.push_back ( _neighborhood_heuristic->get_choice_value () );
				
				// Change internal state due to value selection
				neighborhood_assignment_on_var ( idx );
			}
		} 
	}
	
	// Reset index in neighborhood_heuristic
	_neighborhood_heuristic->set_index ( first_index );
	
	// Extend neighborhood according to the local search strategy 
	for ( auto& idx : index_to_add )
	{
		if ( idx >= 0 )
		{
			_neighborhood.push_back ( idx );
		}
	}
	
	return _neighborhood_val;
}//ls_get_choice_variable
	
	