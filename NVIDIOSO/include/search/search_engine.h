//
//  search_engine.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class represents the interface for a search engine.
//  Different search strategies implement this interface.
//  @note This interface is partially based on the Search interface
//        of the constraint solver JaCoP.
//  @see http://jacop.osolpro.com/ for more details about the JaCoP
//       interface.
//

#ifndef NVIDIOSO_search_engine_h
#define NVIDIOSO_search_engine_h

#include "globals.h"
#include "domain.h"
#include "constraint_store.h"
#include "heuristic.h"
#include "solution_manager.h"
#include "backtrack_manager.h"

class SearchEngine;
typedef std::shared_ptr<SearchEngine> SearchEnginePtr;

class SearchEngine {
protected:
  SearchEngine () {};
  
public:
  virtual ~SearchEngine () {};
  
  /**
   * Set debug option.
   * @param debug_on boolean value indicating if debug
   *        should be enabled.
   */
  virtual void set_debug ( bool debug_on ) = 0;
  
  /**
   * Set debug with trail option.
   * If enabled it prints debug and trail stack behaviours.
   * @param debug_on boolean value indicating if debug
   *        should be enabled.
   */
  virtual void set_trail_debug ( bool debug_on ) = 0;
  
  /**
   * Set a reference to a constraint store.
   * The given store will be used to evaluate
   * the constraints.
   * @param a reference to a constraint store.
   */
  virtual void set_store ( ConstraintStorePtr store ) = 0;
  
  /**
   * Set the heuristic to use to get the variables
   * and the values every time a node of the search
   * tree is explored.
   * @param a reference to a heuristic.
   */
  virtual void set_heuristic ( HeuristicPtr heuristic ) = 0;
  
  /**
   * Set a solution manager for this search engine.
   * @param a reference to a solution manager.
   */
  virtual void set_solution_manager ( SolutionManager* sol_manager ) = 0;
  
  /**
   * Sets a backtrackable manager to this class.
   * @param bkt_manager a reference to a backtrack manager.
   */
  virtual void set_backtrack_manager ( BacktrackManagerPtr bkt_manager ) = 0;
  
  /**
   * Returns the number of backtracks
   * performed by the search.
   * @return the number of backtracks.
   */
  virtual size_t get_backtracks () const = 0;
  
  /**
   * Returns the number of nodes visited
   * by the search.
   * @return the number of visited nodes.
   */
  virtual size_t get_nodes () const = 0;
  
  /**
   * Returns the number of wrong decisions made
   * during the search process.
   * @return the number of wrong decisions.
   * @note a decision is "wrong" depending on the search
   *       engine used to explore the search space.
   *       Usually, a wrong decision is represented by a 
   *       leaf of the search tree which has failed.
   */
  virtual size_t get_wrong_decisions () const = 0;
  
  /**
   * Set maximum number of solutions to be found.
   * @param num_sol the maximum number of solutions.
   * @note  -1 for finding all solutions.
   */
  virtual void set_solution_limit ( size_t num_sol ) = 0;
  
  /**
   * Imposes a timeoutlimit.
   * @param timeout timeout limit.
   * @note -1 for no timeout.
   */
  virtual void set_timeout_limit ( double timeout ) = 0;
  
  /**
   * Sets the time-watcher, i.e., it stores the 
   * computational times of consistency, backtrack, etc.
   * @param watcher_on the boolean value that turns on the
   *        of turns off the time watcher.
   */
  virtual void set_time_watcher ( bool watcher_on ) = 0;
  
  //! Print on standard output last solution found.
  virtual void print_solution () const = 0;
  
  //! Print all solutions found so far
  virtual void print_all_solutions () const = 0;
  
  /**
   * Print on standard output a solutions represented
   * by its index.
   * @param sol_idx the index of the solution to print.
   * @note first solution has index 1.
   */
  virtual void print_solution ( size_t sol_idx ) const = 0;
  
  /**
   * Return the last solution found if any.
   * @return a vector of variables' domains (pointer to)
   *         Each domain is most probably a singleton and together
   *         represent a solution.
   */
  virtual std::vector<DomainPtr> get_solution () const = 0;
  
  /**
   * Return the n^th solution found if any.
   * @param n_sol the solution to get.
   * @return a vector of variables' domains (pointer to)
   *         Each domain is most probably a singleton and together
   *         represent a solution.
   * @note The first solution has index 1.
   */
  virtual std::vector<DomainPtr> get_solution ( int n_sol ) const = 0;
  
  /**
   * It assignes variables one by one.
   * This function is called recursively.
   * @param var the index of the variable (not grounded) to assign.
   * @return true if the solution was found.
   */
  virtual bool label ( int var ) = 0;
  
  /**
   * It performs the actual search.
   * First it sets up the internal items/attributes of search.
   * Then, it calls the labeling function with argument specifying
   * the index of a not grounded variable.
   * @return true if a solution was found.
   */
  virtual bool labeling () = 0;
  
  /**
   * Set a maximum number of backtracks to perform
   * during search.
   * @param the number of backtracks to consider as a limit
   *        during the search.
   */
  virtual void set_backtrack_out ( size_t out_b ) = 0;
  
  /**
   * Set a maximum number of nodes to visit 
   * during search.
   * @param the number of nodes to visit and to be considered
   *        as a limit during the search.
   */
  virtual void set_nodes_out ( size_t out_n ) = 0;
  
  /**
   * Set a maximum number of wrong decisions to make
   * before exiting the search phase.
   * @param the number of wrong decisions to set
   *        as a limit during the search.
   */
  virtual void set_wrong_decisions_out ( size_t out_w ) = 0;
  
  //! Prints info about the search engine
  virtual void print () const = 0;
  
};



#endif
