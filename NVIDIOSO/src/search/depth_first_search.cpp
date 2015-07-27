//
//  depth_first_search.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/08/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "depth_first_search.h"
#include "int_variable.h"

using namespace std;

size_t DepthFirstSearch::_search_id = 0;

DepthFirstSearch::DepthFirstSearch () {
  _search_id++;
  init_search ();
}//DepthFirstSearch

DepthFirstSearch::~DepthFirstSearch () {
  delete _solution_manager;
}//~DepthFirstSearch

void
DepthFirstSearch::init_search () {
  _dbg                 = "DepthFirstSearch - ";
  _depth               = 0;
  _peak_depth          = 0;
  _num_backtracks      = 0;
  _num_nodes           = 0;
  _num_wrong_decisions = 0;
  _debug               = false;
  _trail_debug         = false;
  _time_watcher        = false;
  _search_out          = false;
  _backtrack_out_on    = false;
  _backtracks_out      = 0;
  _nodes_out_on        = false;
  _nodes_out           = 0;
  _wrong_out_on        = false;
  _wrong_out           = 0;
  _timeout_out_on      = false;
  _timeout_out         = -1;
  _store               = nullptr;
  _heuristic           = nullptr;
  _solution_manager    = nullptr;
  _backtrack_manager   = nullptr;
}//init_search

void
DepthFirstSearch::set_debug ( bool debug_on ) 
{
  _debug = debug_on;
}//set_debug

void
DepthFirstSearch::set_trail_debug ( bool debug_on ) 
{
	if ( debug_on )
	{
  		_debug = debug_on;
  	}
  _trail_debug = debug_on;
}//set_debug

void
DepthFirstSearch::set_store ( ConstraintStorePtr store ) {
  _store = store;
}//set_store

void
DepthFirstSearch::set_heuristic ( HeuristicPtr heuristic ) {
  _heuristic = heuristic;
}//set_heuristic

void
DepthFirstSearch::set_solution_manager ( SolutionManager* sol_manager ) {
    if ( sol_manager == nullptr ) return;
    _solution_manager = sol_manager;
}//set_solution_manager

void
DepthFirstSearch::set_backtrack_manager ( BacktrackManagerPtr bkt_manager ) {
  if ( bkt_manager == nullptr ) return;
  
  _backtrack_manager = bkt_manager;
}//set_backtrack_manager

size_t
DepthFirstSearch::get_backtracks () const {
  return _num_backtracks;
}//get_backtracks

size_t
DepthFirstSearch::get_nodes() const {
  return _num_nodes;
}//get_nodes

size_t
DepthFirstSearch::get_wrong_decisions () const {
  return _num_wrong_decisions;
}//get_wrong_decisions

void
DepthFirstSearch::set_timeout_limit ( double timeout )
{
    if ( timeout >= 0 )
    {
        _timeout_out_on = true;
    
        timeval time_stats;
        gettimeofday( &time_stats, NULL );
        _timeout_out = time_stats.tv_sec + (time_stats.tv_usec/1000000.0) + timeout;
    }
}//set_timeout_limit

void
DepthFirstSearch::set_time_watcher ( bool watcher_on )
{
    _time_watcher = watcher_on;
}//set_time_watcher

void
DepthFirstSearch::set_solution_limit ( size_t num_sol )
{
	_solution_manager->set_solution_limit( (int) num_sol );
}//set_solution_limit

std::vector<DomainPtr>
DepthFirstSearch::get_solution () const {
  throw "Not yet implemented";
}//get_solution

std::vector<DomainPtr>
DepthFirstSearch::get_solution ( int n_sol ) const {
  throw "Not yet implemented";
}//get_solution

void
DepthFirstSearch::set_backtrack_out ( size_t out_b )
{
    if ( out_b > 0 )
    {
        _backtracks_out   = out_b;
        _backtrack_out_on = true;
    }
}//set_backtrack_out

void
DepthFirstSearch::set_nodes_out( size_t out_n )
{
    if ( out_n > 0 )
    {
        _nodes_out    = out_n;
        _nodes_out_on = true;
    }
}//set_nodes_out

void
DepthFirstSearch::set_wrong_decisions_out ( size_t out_w )
{
    if ( out_w > 0 )
    {
        _wrong_out    = out_w;
        _wrong_out_on = true;
    }
}//set_wrong_decisions_out

bool
DepthFirstSearch::search_out ()
{
    if ( _search_out ) return true;
  
    if ( _timeout_out_on ) {
        timeval time_stats;
        gettimeofday( &time_stats, NULL );
        double current_time = time_stats.tv_sec + (time_stats.tv_usec/1000000.0);
        if ( current_time >= _timeout_out ) 
        {
            LogMsg << _dbg + "Terminated: timeout reached" << endl;
            _timeout_out = 0;
            _search_out  = true;
        }
    }
    if ( _nodes_out_on && _num_nodes > _nodes_out ) 
    {
    	ostringstream s;
    	s << _depth;
    	
    	LogMsg << _dbg +
                  "Terminated: limit on the number of nodes reached - " +
                  "Depth: " + s.str() << endl;
                  
        _search_out = true;
        return true;
    }
    if ( _wrong_out_on && _num_wrong_decisions > _wrong_out ) 
    {
	ostringstream s;
        s << _depth;
    
        LogMsg << _dbg +
            "Terminated: limit on the number of wrong decisions reached." +
            "Depth: " + s.str() << endl;
              
        _search_out = true;
        return true;
    }
    if ( _backtrack_out_on && _num_backtracks > _backtracks_out ) 
    {
      ostringstream s;
      s << _depth;
    
      LogMsg << _dbg +
          "Terminated: limit on the number of backtracks reached." +
          "Depth: " + s.str() << endl;
      
      _search_out = true;
      return true;
    }
    
    return false;
}//search_out

bool
DepthFirstSearch::labeling ()
{  
  // Base case: no store implies that the model is trivially satisfied.
  if ( _store == nullptr ) return true;
  
  if ( _backtrack_manager == nullptr ) {
    throw NvdException( (_dbg + "No Backtrack Manager").c_str() );
  }
  
  if ( _solution_manager == nullptr ) {
    throw NvdException( (_dbg + "No Solution Listener").c_str() );
  }
  
  /*
   * Initial consistency on the store.
   * @note Exit asap if the problem is unsatisfiable and
   * this can be found with a single propagation, e.g., checking domains.
   */
  bool search_consistent  = _store->consistency();
  
  /*
   * Store the state before any labeling on FD vars.
   * This is done to preserve the variables (status) for further operations
   * on them after the labeling has been completed.
   */
  _backtrack_manager->set_level( _backtrack_manager->get_level () + 1 );
  _depth = _backtrack_manager->get_level ();
  
  if ( _trail_debug ) 
  {
    cout << "Trailstack before labeling:\n";
    _backtrack_manager->print();
  }
  
  if ( search_consistent ) 
  {
    try 
    {
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
    	
    	// Start exploring the search tree from level 0
		search_consistent = label( 0 );
    }
    catch ( NvdException& e ) 
    {
      throw e;
    }
  }
   
  // Reset all the status of the variables 
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
  else {
    cout << "----- Unsatisfiable -----\n";
  }
  
  print ();
  
  return search_consistent;
}//labeling

bool
DepthFirstSearch::label( int var_idx ) 
{
  if ( search_out() ) return false;
  
  Variable * var;
  int value = 0;
  bool consistent;
  
  // Search nodes begins here
  _num_nodes++;
  
  // Checks whether the current store is consistent
  if ( _time_watcher )
  {
    statistics.set_timer( Statistics::TIMING::FILTERING );
  }
  
  consistent = _store->consistency ();
  
  if ( _time_watcher )
  {
    statistics.stopwatch_and_add ( Statistics::TIMING::FILTERING );
  }
  
  if ( !consistent ) 
  {
    if ( _debug )
    {
      LogMsg << _dbg << "Store not consistent at depth " << _depth << endl;
    }
    
    /*
     * There is a leaf representing 
     * a wrong decision (i.e., failed leaf).
     */
    _num_wrong_decisions++;
    return false;
  }
  else 
  {
    // Consistent
    _backtrack_manager->force_storage ();
    _backtrack_manager->set_level ( ++_depth );
    if ( _depth > _peak_depth ) _peak_depth = _depth;
    
    if ( _trail_debug ) 
    {
    	cout << "TrailStack after consistency at level " << _depth << ":\n";
      	_backtrack_manager->print();
    }
    
    var = _heuristic->get_choice_variable ( var_idx );
    if ( var != nullptr ) 
    {
      
      if ( _debug )
      {
    	LogMsg 
    	<< _dbg << "Label V_" << var->get_id() << " (" 
    	<< var->get_str_id() << ")" << " at level " << _depth 
    	<< endl;
       }
        
      try 
      {
    	value = _heuristic->get_choice_value ();
      } 
      catch ( NvdException& e ) 
      {
        throw e;
      }
      
      if ( _debug )
      {
    	LogMsg << _dbg << "-- Label = " << value << " -- " << endl;
	  }

      /*
       * Here it comes the actual labeling.
       * @note care must be taken for non int variables (e.g. set, float).
       * @note it automatically notifies the attached store.
       */
      if ( var->domain_iterator->is_numeric () )
        (static_cast<IntVariable*>(var))->shrink ( value, value );   
        
    }
    else 
    {
      
      // Solution found, so this is not a search node
      if ( _debug )
      {
        LogMsg << _dbg << "Solution found at level " << _depth << endl;
      }
      
      bool cut_search; 
      try 
      {
        cut_search = _solution_manager->notify ();
      } 
      catch ( NvdException& e ) 
      {
        throw e;
      }
  
      _num_nodes--;
      
      if ( _time_watcher )
        statistics.set_timer ( Statistics::TIMING::BACKTRACK );
      
      _backtrack_manager->remove_level ( _depth   );
      _backtrack_manager->set_level    ( --_depth );
      
      if ( _time_watcher )
      {
        statistics.stopwatch_and_add ( Statistics::TIMING::BACKTRACK );
      }
      
      if ( _trail_debug ) 
      {
        cout << "Trailstack after solution has been found at level " << _depth <<
        " (after pop):\n";
        
        _backtrack_manager->print();
        _solution_manager->print_variables ();
      }
      
      return cut_search;
    }
    
    // Recursive call
    int next_index = _heuristic->get_index();
    
    if ( _debug )
    {
      LogMsg << _dbg << "Recursive call from level " << _depth << endl;
    }
    
    try 
    {
      consistent = label ( next_index );
    } 
    catch ( NvdException& e ) 
    {
      throw e;
    }
    
    if ( _debug )
    {
      LogMsg << _dbg << "Return from recursive call at level " << _depth << endl;
    }
    
    // If children are consistent, done exit
    if ( consistent ) 
    {
      var = nullptr;

      if ( _time_watcher )
      {
        statistics.set_timer ( Statistics::TIMING::BACKTRACK );
      }
      
      _backtrack_manager->remove_level ( _depth   );
      _backtrack_manager->set_level    ( --_depth );
      
      if ( _time_watcher )
      {
        statistics.stopwatch_and_add ( Statistics::TIMING::BACKTRACK );
      }
      
      return true;
    }
    else 
    {
      
      /*
       * The current assignment of value to var leads to 
       * a failure somewhere in the subtree, this value 
       * must be removed from the domain of var and another
       * labeling must be called recursively on the same var.
       */
      
      if ( _debug )
      {
        LogMsg << _dbg << "Backtrack on V_" << var->get_id() 
        << " from level " << _depth 
        << endl;
	  }
	  
      if ( _time_watcher )
      {
        statistics.set_timer ( Statistics::TIMING::BACKTRACK );
      }
      
      _backtrack_manager->remove_level ( _depth );
      
      if ( _time_watcher )
      {
        statistics.stopwatch_and_add ( Statistics::TIMING::BACKTRACK );
      }
      
      if ( _trail_debug ) 
      {
        cout << "Trailstack after pop:\n";
        _backtrack_manager->print();
        _solution_manager->print_variables ();
      }
      
      if ( !var->is_singleton() ) 
      {
    	if ( _debug )
        {
          cout << _dbg << "V_" << var->get_id() 
          << " new labeling (rec. call) " << "D^V_" << var->get_id() 
          << "\\{" << value << "}" 
          << endl;
        }
        
        _backtrack_manager->set_level ( _backtrack_manager->get_level() );
        
        /*
         * Avoid considering the value that lead to a failure.
         * @note care must be taken for non int variables (e.g. set, float).
         * @note it automatically notifies the attached store.
         */
        if ( var->domain_iterator->is_numeric () )
        {
          (static_cast<IntVariable*>(var))->subtract ( value );
        }
        
        _backtrack_manager->force_storage ();
        
        if ( _debug )
        {
          if ( var->is_singleton() )
          {
            cout << _dbg << "V_" << var->get_id() << " became assigned with value " 
            << (static_cast<IntVariable*>(var))->min() 
            << endl;
          }
		}
		
        try 
        {
          /*
           * Try to label again here.
           * @note the var_idx remains the same since the search
           *       is still considering the same level of the tree.
           * @note most probably the same variable will be considered in the next
           *       iteration, but the value "value" that led to failure 
           *       is not present anymore in its domain. 
           */		
          consistent = label ( var_idx );
        } 
        catch ( NvdException& e ) 
        {
          throw e;
        }
        
        if ( _debug )
        {
          cout << _dbg << "Return from new labeling of V_" << var->get_id() 
          << " (rec. call) at level " << _depth 
          << endl;
        }
        
        if ( !consistent ) 
        {
          _num_backtracks++;
          
          if ( _time_watcher )
          {
            statistics.set_timer ( Statistics::TIMING::BACKTRACK );
          }
          
          _backtrack_manager->remove_level ( _depth );
          
          if ( _time_watcher )
          {
            statistics.stopwatch_and_add ( Statistics::TIMING::BACKTRACK );
          }
          
          if ( _trail_debug ) 
          {
            cout << "Trailstack after pop:\n";
            _backtrack_manager->print();
            _solution_manager->print_variables ();
          }
        }
      }
      else 
      {
        // Var was singleton, can't label it anymore
        if ( _debug )
        {
          LogMsg << _dbg << "V_" << var->get_id() 
          << " is singleton - fail" 
          << endl;
		}
        var = nullptr;
        consistent = false;
      }
      
      _backtrack_manager->set_level ( --_depth );

      if ( consistent ) return true;
      else              return false;
    }
  }
}//label

void
DepthFirstSearch::print_solution () const {
  _solution_manager->print_solution();
}//print_solution

void
DepthFirstSearch::print_all_solutions () const {
  auto solutions = _solution_manager->get_all_solutions();
  for ( auto sol : solutions )
    cout << sol << endl;
}//print_solution

void
DepthFirstSearch::print_solution ( size_t sol_idx ) const {
  if ( sol_idx < 1 || sol_idx > _solution_manager->number_of_solutions() ) {
    return;
  }
  cout << _solution_manager->get_solution ( sol_idx ) << endl;
}//print_solution

void
DepthFirstSearch::print () const {
  cout << "DepthFirstSearch:\n";
  cout << "Summary\n";
  cout << "\tSolutions:       " << _solution_manager->number_of_solutions () << endl;
  cout << "\tPropagators:     " << _store->num_constraints () << endl;
  cout << "\tPropagations:    " << _store->num_propagations () << endl;
  cout << "\tExplored nodes:  " << get_nodes( ) << endl;
  cout << "\tBacktracks:      " << get_backtracks () << endl;
  cout << "\tWrong decisions: " << get_wrong_decisions () << endl;
  cout << "\tPeak depth:      " << _peak_depth << endl;
  if ( _search_out ) {
    cout << "Search aborted ";
    if ( _timeout_out == 0 )
      cout << "\tTimeout reached." << endl;
    else if ( _nodes_out_on && _num_nodes > _nodes_out )
      cout << "\tMaximum number of nodes reached." << endl;
    else if ( _wrong_out_on && _num_wrong_decisions > _wrong_out )
      cout << "\tMaximum number of wrong decisions reached." << endl;
    else if ( _backtrack_out_on && _num_backtracks > _backtracks_out )
      cout << "\tMaximum number of backtracks reached." << endl;
  }
  _heuristic->print();
}//print


