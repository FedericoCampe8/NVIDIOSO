//
//  simple_local_search.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/22/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a simple local search strategy.
//  It doesn't implement any particular local search algorithm but it defines
//  the framework on which a local search will operate.
//  Different local search strategies are defined by the given Heuristic.
//

#ifndef __NVIDIOSO__simple_local_search__
#define __NVIDIOSO__simple_local_search__

#include "local_search.h"
#include "soft_constraint_store.h"
#include "simple_search_out_manager.h"
#include "neighborhood_backtrack_manager.h"

class SimpleLocalSearch : public LocalSearch {
protected:

	//! Id for this search engine
  	static size_t _search_id;
  
  	std::string _dbg;
  
	//! Stores the number of search nodes explored during search.
  	size_t _num_nodes;
  
  	/**
   	 * Stores the number of wrong decisions that have been made
   	 * during local search. A wrong decision is represented by 
   	 * an assignment of variables which does not satisfy hard constraints.
   	 */
  	std::size_t _num_wrong_decisions;
  	
  	//! Number of iterative improving steps performed
  	std::size_t _II_steps;
  	
  	//! Number of restarts performed
  	std::size_t _restarts;
  	
  	//! Maximum number of restarts to perfrom
  	std::size_t _restarts_out;
  
  	//! Specifies if debug option is on.
  	bool _debug;
  
  	//! Specifies if debug and trail debug options are on.
  	bool _trail_debug;
  
  	//! Specifies if the time-watcher is on
  	bool _time_watcher;
  
  	//! Pointer to the constraint store to use during this search.
  	SoftConstraintStoreSPtr _store;
  
  	//! Pointer to the current heuristic to use during search.
  	LSHeuristicSPtr _ls_heuristic;
  
  	//! Pointer to the current backtrack manager.
  	NeighborhoodBacktrackManagerSPtr _backtrack_manager;
  	
  	//! Pointer to local search initializer
  	SearchInitializerUPtr _ls_initializer;
  	
  	//! Pointer to search out manager
  	SimpleSearchOutManagerSPtr _ls_search_out_manager;
  	
  	//! Pointer to solution manager
  	SolutionManager* _solution_manager;
  	
  	//! Initialization function for member variables
  	virtual void init_search_parameters ();
  	
  	/**
  	 * Backtrack on variable idx.
  	 * @param idx index (0, 1, 2, ...) of the variable to backtrack on.
  	 */
  	virtual void unset_neighborhood ( int idx );
  	
  	/**
  	 * Backtrack on variables in neighborhood.
  	 * @param neighborhood array of the variables to backtrack on.
  	 */
  	virtual void unset_neighborhood ( std::vector<int>& neighborhood );
  	
  	/**
  	 * Notify heuristic that something during the search has changed.
  	 * For example, this method can be used to update the objective values
  	 * into the heuristic after each propagation step.
  	 */ 
  	virtual void notify_heuristic ();
  	 
public:
	SimpleLocalSearch ();
  
  	virtual ~SimpleLocalSearch ();
  
  	/**
  	 * Set debug options.
   	 * @param debug_on boolean value indicating if debug
  	 *        should be enabled.
  	 * @note default debug is off.
   	 */
  	void set_debug ( bool debug_on ) override;
  
  	/**
   	 * Set debug with trail option.
   	 * If enabled it prints debug and trail stack behaviours.
   	 * @param debug_on boolean value indicating if debug
   	 *        should be enabled.
   	 */
  	void set_trail_debug ( bool debug_on ) override;
  
  	/**
   	 * Set a reference to a constraint store.
   	 * The given store will be used to evaluate
   	 * the constraints.
   	 * @param a reference to a constraint store.
   	 */
  	void set_store ( ConstraintStorePtr store ) override;
  
  	/**
   	 * Set the heuristic to use to get the variables
   	 * and the values every time a node of the search
   	 * tree is explored.
   	 * @param a reference to a heuristic.
   	 */
  	void set_heuristic ( HeuristicPtr heuristic ) override;
  
  	/**
   	 * Set a solution manager for this search engine.
   	 * @param a reference to a solution manager.
   	 */
  	void set_solution_manager ( SolutionManager* sol_manager ) override;
   
  	/**
   	 * Sets a backtrackable manager to this class.
   	 * @param bkt_manager a reference to a backtrack manager.
   	 */
  	void set_backtrack_manager ( BacktrackManagerPtr bkt_manager ) override;
  
  	/**
   	 * Set the initializer used to define the initial search positions.
   	 * This defines the initialization of the search process. 
   	 * @param a reference to a local search initializer.
   	 */
  	void set_search_initializer ( SearchInitializerUPtr initializer ) override;
  
  	/**
   	 * Set the manager used by local search algorithms to determine when 
   	 * the search is to be terminated upon reaching a specific search position 
   	 * (and/or memory state).
   	 * @param a reference to a search memory manager object.
   	 */
  	void set_search_out_manager ( SearchOutManagerSPtr search_out_manager ) override;
  
  	/**
   	 * Returns the number of backtracks performed by the search.
   	 * @return the number of backtracks.
   	 * @note by default it returns 0.
   	 */
  	std::size_t get_backtracks () const override;
  
  	/**
     * Returns the number of nodes visited by the search.
   	 * @return the number of visited nodes.
   	 */
  	std::size_t get_nodes () const override;
  
  	/**
     * Returns the number of wrong decisions made during the search process.
  	 * @return the number of wrong decisions.
   	 * @note a decision is "wrong" when the assignment does not satisfy hard constraints.
   	 */
	std::size_t get_wrong_decisions () const override;
  
  	/**
   	 * Set maximum number of solutions to be found.
   	 * @param num_sol the maximum number of solutions.
   	 * @note  this function will change SearchOutManager settings.
   	 */
  	void set_solution_limit ( size_t num_sol ) override;
  
  	/**
   	 * Imposes a timeoutlimit.
   	 * @param timeout timeout limit.
   	 * @note -1 for no timeout.
   	 * @note  this function will change SearchOutManager settings.
   	 */
 	void set_timeout_limit ( double timeout ) override;
  
  	/**
   	 * Sets the time-watcher, i.e., it stores the
   	 * computational times of consistency, backtrack, etc.
   	 * @param watcher_on the boolean value that turns on the
   	 *        of turns off the time watcher.
   	 */
  	void set_time_watcher ( bool watcher_on ) override;
  
  	/**
   	 * Return the last solution found if any.
   	 * @return a vector of variables' domains (pointer to)
   	 *         Each domain is most probably a singleton and together
   	 *         represent a solution.
   	 */
  	std::vector<DomainPtr> get_solution () const override;
  
  	/**
     * Return the n^th solution found if any.
   	 * @param n_sol the solution to get.
   	 * @return a vector of variables' domains (pointer to)
   	 *         Each domain is most probably a singleton and together
   	 *         represent a solution.
   	 * @note The first solution has index 1.
   	 */
  	std::vector<DomainPtr> get_solution ( int n_sol ) const override;
  
  	/**
   	 * It assignes variables one by one.
   	 * This function is called iteratively.
   	 * @param var the index of the variable (not grounded) to assign.
   	 * @return true if the solution was found.
   	 */
  	bool label ( int var ) override;
  
  	/**
  	 * It performs the actual search.
   	 * First it sets up the internal items/attributes of search.
   	 * Then, it calls the labeling function with argument specifying
   	 * the index of a not grounded variable.
   	 * @return true if a solution was found.
   	 */
 	bool labeling () override;
   
  	/**
   	 * Set a maximum number of backtracks to perform during search.
   	 * @note This function does not have any effect.
   	 */
  	void set_backtrack_out ( size_t out_b ) override;
  
  	/**
   	 * Set a maximum number of nodes to visit during search.
   	 * @param the number of nodes to visit and to be considered
   	 *        as a limit during the search.
   	 * @note  this function will change SearchOutManager settings.
   	 */
  	void set_nodes_out ( size_t out_n ) override;
  
  	/**
  	 * Set a maximum number of wrong decisions to make
   	 * before exiting the search phase.
   	 * @param the number of wrong decisions to set
   	 *        as a limit during the search.
   	 * @note  this function will change SearchOutManager settings.
   	 */
  	void set_wrong_decisions_out ( size_t out_w ) override;
  	
  	
  	/**
   	 * Set the maximum number of iterative improving steps to perform.
   	 * An Iterative Improving (II) step is a restart of the (local) search strategy
   	 * where the starting point is the best solution found so far.
   	 * @param ii_limit unsigned value representing the maximum number of 
   	 *        iterative improving steps to perform.
   	 */
  	void set_iterative_improving_limit ( std::size_t ii_limit = 0 ) override;
  	
  	/**
   	 * Set the maximum number of restarts to perform.
   	 * A restarts calles the (local) search strategy again, starting from 
   	 * the initial solution provided to the strategy (e.g., by the 
   	 * search initializer).
   	 * @param restarts_limit unsigned value representing the maximum number of 
   	 *        restarts to perform (default 0 restarts).
   	 * @note A restart is not performed if some other limit has been already reached,
   	 *       e.g., a restart is not performed if timeout limit has been reached.
   	 */
  	void set_restarts_limit ( std::size_t restarts_limit = 0 ) override;
  	
  	
  	//! Print on standard output last solution found.
  	void print_solution () const override;
  
  	//! Print all solutions found so far
 	void print_all_solutions () const override;
  
  	/**
  	 * Print on standard output a solutions represented by its index.
   	 * @param sol_idx the index of the solution to print.
   	 * @note first solution has index 1.
   	 */
  	void print_solution ( size_t sol_idx ) const override;
  
  	//! Prints info about the search engine
  	void print () const override;
};

#endif /* defined(__NVIDIOSO__simple_local_search__) */
