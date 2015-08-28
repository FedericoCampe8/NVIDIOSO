//
//  local_search_heuristic.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/22/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class declares the interface for a local search heuristic.
//  A local search heuristic actually implements a local search strategy
//  by mapping each search position and memory state onto a probability distribution 
//  over its neighboring search positions and memory states — this (implementation of)
//  class specifies what happens in every search step;
//
 
#ifndef __NVIDIOSO__local_search_heuristic__
#define __NVIDIOSO__local_search_heuristic__

#include "globals.h"
#include "heuristic.h"
#include "objective_state.h" 

class LocalSearchHeuristic;
typedef std::unique_ptr<LocalSearchHeuristic> LSHeuristicUPtr;
typedef std::shared_ptr<LocalSearchHeuristic> LSHeuristicSPtr; 

class LocalSearchHeuristic : public Heuristic {	
public:
	 
	virtual ~LocalSearchHeuristic () {}; 
	
	//! Reset the internal state of the local search
	virtual void reset_state () = 0;
	
	/**
	 * Change set of variables where local search is performed on.
	 * @note This function resets the internal state of the search.
	 */
	virtual void set_search_variables ( std::vector< Variable* >& vars, Variable * obj_var = nullptr ) = 0; 
	  
	/**
	 * Update the current objective value with external parameters.
	 * @param num number of constraints which are not satisfied by the current assignment.
	 */
	virtual void update_objective ( std::size_t num_unsat, double unsat_level ) = 0;
	
	/**
	 * Generalize get_index to return a set of indexes.
	 * For better performance, local search strategies 
	 * should use this method instead of get_index().
	 */
	virtual std::vector<int> ls_get_index () const = 0;
	
	/**
	 * Generalize get_choice_variable to return a set of variables.
	 * For better performance, local search strategies should use this method 
	 * instead of get_choice_variable().
	 */
	virtual std::vector<Variable *> ls_get_choice_variable ( std::vector< int > index ) = 0;
	
	/**
	 * Generalize get_choice_value to return a set of values.
	 * For better performance, local search strategies should use this method 
	 * instead of get_choice_value().
	 */
	virtual std::vector<int> ls_get_choice_value () = 0;
};

#endif /* defined(__NVIDIOSO__local_search_heuristic__) */
