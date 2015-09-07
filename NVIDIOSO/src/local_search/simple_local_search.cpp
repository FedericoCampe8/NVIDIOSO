//
//  simple_local_search.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/22/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "simple_local_search.h"
#include "int_variable.h"

using namespace std;

size_t SimpleLocalSearch::_search_id = 0;

SimpleLocalSearch::SimpleLocalSearch () {
  _search_id++;
  init_search_parameters ();
}//SimpleLocalSearch

SimpleLocalSearch::~SimpleLocalSearch () {
	delete _solution_manager;
}//~SimpleLocalSearch

void
SimpleLocalSearch::init_search_parameters () 
{
  _dbg                   = "SimpleLocalSearch - ";
  _num_nodes             = 0;
  _num_wrong_decisions   = 0;
  _II_steps			     = 0;
  _restarts              = 0;
  _restarts_out          = 0;
  _debug                 = false;
  _trail_debug           = false;
  _time_watcher          = false;
  _store                 = nullptr;
  _ls_heuristic          = nullptr;
  _backtrack_manager     = nullptr;
  _ls_initializer        = nullptr;
  _ls_search_out_manager = nullptr;
  _solution_manager      = nullptr;
}//init_search_parameters

void
SimpleLocalSearch::set_debug ( bool debug_on ) 
{
	_debug = debug_on;
}//set_debug

void
SimpleLocalSearch::set_trail_debug ( bool debug_on ) 
{
	if ( debug_on )
	{
  		_debug = debug_on;
  	}
  _trail_debug = debug_on;
}//set_debug

void
SimpleLocalSearch::set_store ( ConstraintStorePtr store ) 
{
	_store = std::dynamic_pointer_cast< SoftConstraintStore > ( store );
}//set_store

void
SimpleLocalSearch::set_heuristic ( HeuristicPtr heuristic ) {
	_ls_heuristic = dynamic_pointer_cast<LocalSearchHeuristic> ( heuristic );
}//set_heuristic

void
SimpleLocalSearch::set_solution_manager ( SolutionManager* sol_manager ) 
{
	if ( sol_manager == nullptr ) return;
    _solution_manager = dynamic_cast<LocalSearchSolutionManager*> ( sol_manager );
}//set_solution_manager
 
void 
SimpleLocalSearch::set_backtrack_manager ( BacktrackManagerPtr bkt_manager ) 
{
	if ( bkt_manager == nullptr ) return;
  	_backtrack_manager = std::dynamic_pointer_cast< NeighborhoodBacktrackManager > ( bkt_manager );
}//set_backtrack_manager

void
SimpleLocalSearch::set_search_initializer ( SearchInitializerUPtr initializer ) 
{
	_ls_initializer = std::move ( initializer );
}//set_search_initializer

void
SimpleLocalSearch::set_search_out_manager ( SearchOutManagerSPtr search_out_manager )
{
	_ls_search_out_manager = std::dynamic_pointer_cast< SimpleSearchOutManager > ( search_out_manager );
}//set_search_out_manager

size_t
SimpleLocalSearch::get_backtracks () const 
{
	return 0;
}//get_backtracks

size_t
SimpleLocalSearch::get_nodes() const 
{
	return _num_nodes;
}//get_nodes

size_t
SimpleLocalSearch::get_wrong_decisions () const 
{
	return _num_wrong_decisions;
}//get_wrong_decisions

void
SimpleLocalSearch::set_timeout_limit ( double timeout )
{
	// Sanity check
	assert ( _ls_search_out_manager != nullptr );
	
	if ( _debug )
    {
    	if ( timeout >= 0.0 )
    	{
    		LogMsg << _dbg << "Timeout set to " << timeout << " sec." << endl;
    	}
    	else
    	{
    		LogMsg << _dbg << "No timeout set." << endl;
    	}
    } 
	_ls_search_out_manager->set_time_out ( timeout );
}//set_timeout_limit

void
SimpleLocalSearch::set_time_watcher ( bool watcher_on )
{
	_time_watcher = watcher_on;
}//set_time_watcher

void
SimpleLocalSearch::set_solution_limit ( size_t num_sol )
{
	// Sanity check
	assert ( _solution_manager != nullptr );
	
	/*
	 * Instead of search_out_manager, 
	 * use solution manager for solution limit.
	 */
	if ( _debug )
    {
    	if ( num_sol > 0 )
    	{
    		LogMsg << _dbg << "Solution limit set to " << num_sol << " solutions." << endl;
    		_solution_manager->set_solution_limit( (int) num_sol );
    	}
    	else
    	{
    		LogMsg << _dbg << "No solutions limit set." << endl;
    		_solution_manager->set_solution_limit( -1 );
    	}
    } 
}//set_solution_limit

std::vector<DomainPtr>
SimpleLocalSearch::get_solution () const 
{
	throw "Not yet implemented";
}//get_solution

std::vector<DomainPtr>
SimpleLocalSearch::get_solution ( int n_sol ) const 
{
	throw "Not yet implemented";
}//get_solution

void
SimpleLocalSearch::set_backtrack_out ( size_t out_b )
{
}//set_backtrack_out

void
SimpleLocalSearch::set_nodes_out( std::size_t out_n )
{
	// Sanity check
	assert ( _ls_search_out_manager != nullptr );
	
	if ( _debug )
    {
    	if ( out_n > 0 )
    	{
    		LogMsg << _dbg << "Nodes out limit set to " << out_n << " nodes." << endl;
    	}
    	else
    	{
    		LogMsg << _dbg << "No nodes out limit set." << endl;
    	}
    } 
	_ls_search_out_manager->set_num_nodes_out ( out_n );
}//set_nodes_out

void
SimpleLocalSearch::set_wrong_decisions_out ( std::size_t out_w )
{
	// Sanity check
	assert ( _ls_search_out_manager != nullptr );
	
	if ( _debug )
    {
    	if ( out_w > 0 )
    	{
    		LogMsg << _dbg << "Wrong decisions limit set to " << out_w << " wrong decisions." << endl;
    	}
    	else
    	{
    		LogMsg << _dbg << "No wrong decisions limit set." << endl;
    	}
    } 
	_ls_search_out_manager->set_num_wrong_decisions_out ( out_w );
}//set_wrong_decisions_out

void  
SimpleLocalSearch::set_iterative_improving_limit ( std::size_t ii_limit )
{
	// Sanity check
	assert ( _ls_search_out_manager != nullptr );
	
	if ( _debug )
    {
    	if ( ii_limit > 0 )
    	{
    		LogMsg << _dbg << "Iterative Improving limit set to " << ii_limit << " steps." << endl;
    	}
    	else
    	{
    		LogMsg << _dbg << "No Iterative Improving limit set." << endl;
    	}
    } 

	_ls_search_out_manager->set_num_iterative_improvings_out ( ii_limit );
}//set_iterative_improving_limit

void 
SimpleLocalSearch::set_restarts_limit ( std::size_t restarts_limit )
{
	// Sanity check
	assert ( _ls_search_out_manager != nullptr );
	
	if ( _debug )
    {
    	if ( restarts_limit > 0 )
    	{
    		LogMsg << _dbg << "Restarts limit set to " << restarts_limit << " restarts" << endl;
    	}
    	else
    	{
    		LogMsg << _dbg << "No restarts limit set." << endl;
    	}
    }  
	_restarts_out = restarts_limit;
	_ls_search_out_manager->set_num_restarts_out ( restarts_limit );
}//set_iterative_improving_limit

void 
SimpleLocalSearch::unset_neighborhood ( int idx )
{
	if ( _debug )
    {
    	LogMsg 
    	<< _dbg << "Unset " << idx+1 << " variable of current neighborhood"
    	<< " at iteration " << _num_nodes 
    	<< endl;
	}
	
	_backtrack_manager->remove_level_on_var ( idx );
}//unset_neighborhood
  	
void 
SimpleLocalSearch::unset_neighborhood ( std::vector<int>& neighborhood )
{
	for ( auto& var : neighborhood )
	{
		unset_neighborhood ( var );
	}
}//unset_neighborhood

void
SimpleLocalSearch::notify_heuristic () 
{
	_ls_heuristic->update_objective ( _store->num_unsat_constraints (), 
  									  _store->get_unsat_level_constraints () );
}//notify_heuristic
	 
bool
SimpleLocalSearch::labeling ()
{  
	// Sanity check (no store -> model satisfied)
	if ( _store == nullptr ) 
	{
		return true;
	}
	else
	{
		/*
		 * Reset the internal state of the constraint store (initialize its internal state).
		 * This is done in order to initialize the store before any propagation.
		 * @note this is a store which handles soft constraints and has an internal state.
		 */
		_store->reset_state (); 
	}
  
  	std::string err_msg;
  	if ( _backtrack_manager == nullptr ) 
  	{
  		err_msg = _dbg + "No Backtrack Manager";
  		LogMsg << err_msg << endl;
    	throw NvdException( err_msg.c_str() );
  	}
  
  	if ( _solution_manager == nullptr ) 
  	{
  		err_msg = _dbg + "No Solution Listener";
  		LogMsg << err_msg << endl;
    	throw NvdException( err_msg.c_str() );
  	}
  	
  	if ( _ls_heuristic == nullptr ) 
  	{
  		err_msg = _dbg + "No Local Search Heuristic has been set";
  		LogMsg << err_msg << endl;
    	throw NvdException( err_msg.c_str() );
  	}
  	
  	if ( _ls_initializer == nullptr ) 
  	{
  		err_msg = _dbg + "No Local Search Initializer has been set";
  		LogMsg << err_msg << endl;
    	throw NvdException( err_msg.c_str() );
  	}
  	
  	if ( _ls_search_out_manager == nullptr ) 
  	{
  		err_msg = _dbg + "No Search Out object has been set";
  		LogMsg << err_msg << endl;
    	throw NvdException( err_msg.c_str() );
  	}
  
  	/*
   	 * Initial consistency on constraints into the store.
   	 * @note Exit asap if the problem is unsatisfiable and
   	 * this can be found with a single propagation, 
   	 * e.g., satisfiability on domains. 
   	 */
  	bool search_consistent = _store->consistency();
  
  	/*
   	 * Store the state before any labeling on FD vars.
   	 * This is done to preserve the variables (status) 
   	 * in order to "unset" variables in the neighborhood later
   	 * and perform local search them.
   	 */
  	_backtrack_manager->set_level ( _backtrack_manager->get_level () + 1 );
  	
  	if ( _trail_debug ) 
  	{
    	std::cout << _dbg << "Trailstack before labeling:\n";
    	_backtrack_manager->print();
  	}
  
  	// Set initial assignment where the local search strategy starts at
  	_ls_initializer->initialize ();
  	
  	if ( _debug )
    {
    	_ls_initializer->print_initialization ();
	}

	/*
	 * Start performing local search.
	 * @note the tree level is not important here.
	 * @note a search may be not consistent if all search space has been 
	 *       explored and no solutions satisfying hard constraints have been found.
	 *       A search may be inconsistent also when the search reached timeout
	 *       (or it has been terminated for some other reasons) and no solutions
	 *   	 satisfying hard constraints have been found.
	 */
	if ( search_consistent ) 
  	{	
  		/*
		 * Reset out manager.
		 * This is done in order to reset all parameters of the manager before
		 * starting the actual search.
		 * For example, this sets the current time to this point of computation
		 * if the timeout_evaluator is activated.
		 */
  		_ls_search_out_manager->reset_out_evaluator ();
		try 
    	{
    		do
    		{// Restarts
    			// II steps for each Restats
    			_II_steps = 0;
    			do 
    			{// Iterative Improvements
    				/*
    		 	 	 * Force storage for variables before any operation on them.
    	 	 	 	 * @note if we don't force storage, it may be the case where the
    	 	 	 	 *       following propagation does not modify any domain and, therefore,
    	 	 	 	 *       no variable notifies backtrack manager.
    	 	 	 	 *       The following labeling will notify the backtrack manager which 
    	 	 	 	 *       will store a modified domain (i.e., the singleton) without storing
    	 	 	 	 *       the actual domain of the labeled variable.
    	 	 	 	 */ 
    				_backtrack_manager->force_storage (); 
    		
    				// Reset the internal state of the store before every local search iteration
    				_store->reset_state ();
    		
    				// Reset the internal state of the heuristic before every local search iteration
    				_ls_heuristic->reset_state ();
    		
					search_consistent = label ( 0 );
				 
					_ls_search_out_manager->upd_iterative_improvings_steps ( ++_II_steps );
				}
				while ( !_ls_search_out_manager->search_out () );
				
				// Avoid restoring states if no Restart has to be performed
				if ( _restarts + 1 < _restarts_out )
				{
					// Unlabel variables
					_backtrack_manager->remove_level( _backtrack_manager->get_level () );
				
					// Store the current unlabeled variables 
					_backtrack_manager->set_level ( _backtrack_manager->get_level () + 1 );
				
					// Reset initial solution
					_ls_initializer->initialize_back ();
				}
			}//_restarts
			while ( ++_restarts < _restarts_out );
		}
		catch ( NvdException& e ) 
    	{
      		throw e;
    	}
	}
	
	// Reset all variables (unlabel variables)
  	_backtrack_manager->remove_level( _backtrack_manager->get_level() );
  
  	// Print solutions and info about the search
  	if ( search_consistent || _solution_manager->number_of_solutions() ) 
  	{
		if ( _solution_manager->number_of_solutions() == 1 )
		{
    		cout << "----- Solution -----\n";
    	}
    	else
    	{
    		cout << "----- Solutions -----\n";
    	}
    	print_all_solutions ();
    	cout << "---------------------\n";
  	}
  	else 
  	{
    	cout << "----- Unsatisfiable -----\n";
  	}
  
	print ();
  
	return search_consistent;
}//labeling

bool
SimpleLocalSearch::label ( int var_idx ) 
{
	std::vector<int> idxs_neighborhood = _ls_heuristic->ls_get_index ();

  	int value {};
  	bool consistent {};
  	std::vector<int> vals_neighborhood;
  	std::vector<Variable *> vars_neighborhood;
  	bool terminate = (idxs_neighborhood.size() == 0);
	while ( !terminate )
	{
		unset_neighborhood ( idxs_neighborhood );
		
		// Search nodes begins here
  		_num_nodes++;
  		vars_neighborhood = _ls_heuristic->ls_get_choice_variable ( idxs_neighborhood );
  		if ( vars_neighborhood.size() != 0 ) 
    	{
    		if ( _debug )
      		{
      			for ( auto& var : vars_neighborhood )
      			{
    				LogMsg 
    				<< _dbg << "Label V_" << var->get_id() << " (" 
    				<< var->get_str_id() << ")" << " at iteration " << _num_nodes 
    				<< endl;
    			}
       		}
       		
       		try 
      		{
    			vals_neighborhood = _ls_heuristic->ls_get_choice_value ();
      		} 
      		catch ( NvdException& e ) 
      		{
        		throw e;
      		}
       		
       		// Sanity check
       		assert ( vals_neighborhood.size () == vars_neighborhood.size() );
       		if ( _debug )
      		{
      			for ( auto& value : vals_neighborhood )
      			{
    				LogMsg << _dbg << "-- Label = " << value << " -- " << endl;
    			}
	  		}
	  		
	  		/*
       		 * Here it comes the actual labeling.
       		 * @note care must be taken for non int variables (e.g. set, float).
       		 * @note it automatically notifies the attached store.
       		 */
			int val_idx {};
			for ( auto& var : vars_neighborhood )
			{
      			if ( var->domain_iterator->is_numeric () )
      			{
        			(static_cast<IntVariable*>(var))->shrink ( vals_neighborhood [ val_idx ], vals_neighborhood [ val_idx ] );
        		}
        		++val_idx;
	  		}
    	}
    	else
    	{ 
    		/*
    		 * Solution found, so this is not a search node.
    		 * Store this solution in the solution manager along with its value.
    		 * Solution manager will keep the k best solutions found so far.
    		 */
      		if ( _debug )
      		{
        		LogMsg << _dbg << "Solution found at iteration " << _num_nodes << endl;
      		}
      
      		try 
      		{
      			//@todo Check on epsilon, sat constraints, etc.
        		terminate = _solution_manager->notify ();
        		if ( terminate )
        		{
        			_ls_search_out_manager->force_out ();
        		}
      		} 
      		catch ( NvdException& e ) 
      		{
        		throw e;
      		}
  
      		_num_nodes--;
    	}
    	
    	// Checks whether the current store is consistent
  		if ( _time_watcher )
  		{
    		statistics.set_timer( Statistics::TIMING::FILTERING );
  		}
  
  		// Constraint propagation
  		consistent = _store->consistency ();
  		
  		if ( _time_watcher )
  		{
    		statistics.stopwatch_and_add ( Statistics::TIMING::FILTERING );
  		}
  
  		if ( !consistent ) 
  		{
    		if ( _debug )
    		{
      			LogMsg << _dbg << "Store not consistent at iteration " 
      			<< _num_nodes << endl;
    		}
    
     		// There is a leaf representing a wrong decision (i.e., failed leaf).
    		_ls_search_out_manager->upd_wrong_decisions ( ++_num_wrong_decisions );
    		
    		continue;
  		}
  		 
		if ( _debug )
      	{	 
        	LogMsg << _dbg << "Number of unsatisfied constraints at iteration " 
        	<< _num_nodes << ": " << _store->num_unsat_constraints () << endl;
        	
        	LogMsg << _dbg << "Unsatisfiability level at iteration " 
        	<< _num_nodes << ": " << _store->get_unsat_level_constraints () << endl;
      	}
  		
  		/*
  		 * Notify the heuristic that labeling has been performed and
  		 * constraint propagation has propagated constraints.
  		 * This will update some values (e.g., number of unsat constraints) in heuristic.
  		 */
  		notify_heuristic ();
  		
  		/*
  		 * Update solution manager on values changed due to propagation.
  		 * @note notifies solution manager that propagation has been performed and
  		 *       hence number of constraints and objective value could be both changed.
  		 * @note out manager is notified as soon as possible after propagation.
  		 *       Other labelings could produce worst values on the objective.
  		 */
  		terminate = _solution_manager->notify_on_propagation ( _store->num_unsat_constraints () );
  		 
  		// Proceed with local search on a different neighborhood
  		idxs_neighborhood = _ls_heuristic->ls_get_index ();
  		
  		// Local search has finished to explore the neighborhood
  		if ( idxs_neighborhood.size() == 0 )
  		{
  			// Solution found, so this is not a search node
      		if ( _debug )
      		{
        		LogMsg << _dbg << "Solution found at iteration " << _num_nodes << endl;
      		}
      
      		try 
      		{	
        		terminate = _solution_manager->notify ();
        		if ( terminate )
        		{
        			_ls_search_out_manager->force_out ();
        		}
      		} 
      		catch ( NvdException& e ) 
      		{
        		throw e;
      		}
  
      		_num_nodes--;
      		
      		terminate = true;
  		}
  		
  		// Update total number of nodes explored
  		_ls_search_out_manager->upd_nodes ( _num_nodes );
  		
	}//while

	return true;
}//label

void
SimpleLocalSearch::print_solution () const 
{
	_solution_manager->print_solution();
}//print_solution

void
SimpleLocalSearch::print_all_solutions () const 
{
	auto solutions = _solution_manager->get_all_solutions();
  	for ( auto sol : solutions )
  	{
    	cout << sol << endl;
    }
}//print_solution

void
SimpleLocalSearch::print_solution ( size_t sol_idx ) const 
{
	if ( sol_idx < 1 || sol_idx > _solution_manager->number_of_solutions() ) 
  	{
		return;
  	}
  	cout << _solution_manager->get_solution ( sol_idx ) << endl;
}//print_solution

void
SimpleLocalSearch::print () const 
{
	cout << "LocalSearch:\n";
  	cout << "Summary\n";
  	cout << "\tSolutions:                 " << _solution_manager->number_of_solutions () << endl;
  	cout << "\tPropagators:               " << _store->num_constraints () << endl;
  	cout << "\tPropagations:              " << _store->num_propagations () << endl;
  	cout << "\tExplored nodes:            " << get_nodes( ) << endl;
  	cout << "\tWrong decisions:           " << get_wrong_decisions () << endl;
	cout << "\tRestarts:                  " << _restarts_out << endl;
  	cout << "\tIterative improving steps: " << _II_steps << endl;
 	_ls_heuristic->print();
}//print


