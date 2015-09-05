//
//  simple_local_search_heuristic.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/23/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a general abstract class for a simple local search heuristic.
//

#ifndef __NVIDIOSO__simple_local_search_heuristic__
#define __NVIDIOSO__simple_local_search_heuristic__

#include "local_search_heuristic.h"
#include "neighborhood_heuristic.h"
#include "neighborhood_evaluator.h"
#include "search_memory_manager.h"

class SimpleLocalSearchHeuristic : public LocalSearchHeuristic {
private:
	
	// Internal index used to override functions from Heuristic class
	mutable int _current_index;

	// Auxiliary array of neighborhood indexes, keep for efficiency reasons
	mutable std::vector < int > _neighborhood_idx;
	
	// Auxiliary array of neighborhood values, keep for efficiency reasons
	mutable std::vector < int > _neighborhood_val; 
	
	// Auxiliary array of neighborhood variables, keep for efficiency reasons
	mutable std::vector<Variable *> _neighborhood_var;
	 
protected:
	
	//! Debug information string
  	std::string _dbg;
  	
  	/**
  	 * Pointer to the objective variable, i.e., the variable which value represents
  	 * the result of the objective function.
  	 * @note this variable may be NULL depending on the model to solve.
  	 */
  	const Variable * _obj_variable;
  	
	//! The array of (pointers to) variables on where to perform local search. 
  	std::vector< Variable* > _fd_variables;
  	
  	/**
	 * Pointer to the actual heuristic/local search strategy.
	 * This pointer points to an instance of a heuristic class
	 * implementing local search strategy such as Hill Climbing,
	 * Simulated Annealing, Iterated Conditional Modes, Gibbs Sampling, etc.
	 * @note the combination of local search heuristic and neighborhood evaluator 
	 * defines completely a local search strategy.
	 * @note this determines the positions that can be reached in one search step 
	 *       at any given time during the search process.
	 */
	NeighborhoodHeuristicUPtr _neighborhood_heuristic;
	
	/**
	 * Pointer to a neighborhood evaluator class which selects the actual
	 * label for a given variable. 
	 * This class takes the memory of the search and defines which is the best
	 * label according to a objective value parameter.
	 * @note the combination of local search heuristic and neighborhood evaluator 
	 * defines completely a local search strategy.
	 */ 
	NeighborhoodEvaluatorUPtr _neighborhood_evaluator;
	
	  	
  	/**
   	 * Set the search memory manager used by some local search algorithms (e.g., Tabu search).
   	 * The search memory manager holds information about the state of the search mechanism 
   	 * beyond the search position (e.g., tabu tenure values in the case of tabu search); 
   	 * @param a reference to a search memory manager object.
   	 */
  	MemoryManagerUPtr _memory_manager; 
  	
	/** 
	 * Ordered array of indexes of the variables currently considered
	 * by the local search strategy w.r.t. the
	 * variable in _fd_variables.
	 * This array represents a meta-indexing array for the get_index() function.
	 * @note The name "neighborhood" usually refers to the possible values reachable 
	 *       from a given point of the search strategy. 
	 *       Here we mean the set of variables currently under consideration by the
	 *       local search strategy.
	 */
	 std::vector< int > _neighborhood; 
	  
	/**
	 * Lookup table where key = index (0, 1, 2, ...) of variables 
	 * in _fd_variables and value Boolean value stating whether 
	 * a given variable has been fully explored/considered by 
	 * the current local search strategy.
	 * The variable idx is "fully explored" if the local search strategy 
	 * has determined a value for idx.  
   	 */
  	std::unordered_map< int, bool > _explored_variables;
   
  	//! Reset the _explored_variables table to false
	void reset_explored_variables ();
	 
	/**
	 * Return true if the variable at index idx in  is fully explored, false otherwise.
	 * @param idx index in _fd_variables.
	 * @return _explored_variables [ idx ]
	 */
	bool is_explored_var ( int idx ) const;
	
	//! Set variable at index idx as explored or not depending on the value of val
	void set_explored_var ( int idx, bool val = true );
	
	/**
	 * This method is to design specific local search strategies.
	 * It is invoked when the client ask for values by invoking the 
	 * (ls_)get_choice_value method.
	 * By implementing this method a specific local search strategy can 
	 * decide, for example, when a complete exploration of a given variable idx
	 * has been performed.
	 * @param var_index, index of the variable in _fd_variables under consideration.
	 * @todo Use observer pattern.
	 */
	virtual void notify_on_var_assignment ( int var_index ) = 0;
	
	/**
	 * This method returns true if the local search strategy 
	 * is performing the last assignment on variable at index "var_index".
	 * In general, a local search strategy may have to try multiple labelings
	 * on the same variable before deciding which is the best value to assign to it
	 * (e.g., Iterated Conditional Modes).
	 * This method can return always false and the search strategy will consider
	 * only neighborhood_assignment_complete() but the correct implementation of
	 * this method can speedup the local search strategy by "preparing" the next
	 * variable to label at the same time the current variable is being assigned.
	 * @param var_index index of the variable in _fd_variables to query. 
	 * @return true if the current assignment for var_index is the last one, 
	 *         false otherwise.
	 * @note For example, Iterated Conditional Modes returns true only on the
	 *       last element of a given variable, while Hill Climbing returns
	 *       true after the first assignment.
	 */
	virtual bool neighborhood_last_assignment_on_var ( int var_index ) = 0;
	
	/**
	 * This methods returns true if the local search strategy has already
	 * considered the variable at index var_index in _fd_variables and it is
	 * now ready to consider a new variable or to terminate the search.
	 * @param var_index index of the variable in _fd_variables to query.
	 * @return true if _fd_variables[var_index] has been already considered 
	 *         by the local search strategy for its assignment.
	 */
	virtual bool neighborhood_complete_assignment_on_var ( int var_index ) = 0;
	
	/** 
	 * This method returns the set of variable indexes corresponding to
	 * the starting neighborhood set, i.e., the variables tin _fd_variables
	 * to free in order to start the local search strategy.
	 */
	virtual std::vector<int> starting_neighborhood () = 0;
	
	//! Constructor
	SimpleLocalSearchHeuristic ( std::vector< Variable* > vars, Variable * obj_var = nullptr );
	
public:
	
	virtual ~SimpleLocalSearchHeuristic ();
	
	//! Reset the internal state of the local search
	void reset_state () override;
	
	/**
	 * Change set of variables where local search is performed on.
	 * @param vars array of pointer to the variables to label to change.
	 * @param obj_var pointer to the objective variable to change (by default not changed).
	 * @note This function resets the internal state of the search.
	 */
	void set_search_variables ( std::vector< Variable* >& vars, Variable * obj_var = nullptr ) override; 
	 
	/**
	 * Update the current objective value with external parameters.
	 * @param num_unsat number of constraints which are not satisfied by the current assignment.
	 * @param unsat_level level of unsatisfiability of the current assignment.
	 */
	void update_objective ( std::size_t num_unsat, double unsat_level ) override;
	 
	int get_index () const override;
	
	Variable * get_choice_variable ( int index ) override;
	
	int get_choice_value () override;
	
	/**
	 * Generalize get_index to return a set of indexes.
	 * For better performance, local search strategies should use this method 
	 * instead of get_index().
	 */
	std::vector<int> ls_get_index () const;
	
	/**
	 * Generalize get_choice_variable to return a set of variables.
	 * For better performance, local search strategies should use this method 
	 * instead of get_choice_variable().
	 */
	std::vector<Variable *> ls_get_choice_variable ( std::vector< int > index );
	
	/**
	 * Generalize get_choice_value to return a set of values.
	 * For better performance, local search strategies should use this method 
	 * instead of get_choice_value().
	 */
	std::vector<int> ls_get_choice_value (); 
};

#endif /* defined(__NVIDIOSO__simple_local_search_heuristic__) */
