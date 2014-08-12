//
//  depth_first_search.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class implements depth first search.
//  @note It ensures backtrack and a depth first visit of the
//        search tree. Different heuristic may be choosen for
//        exploring variables (i.e., variable order) and values
//        (i.e., value assignment).
//

#ifndef __NVIDIOSO__depth_first_search__
#define __NVIDIOSO__depth_first_search__

#include "search_engine.h"

class DepthFirstSearch : public SearchEngine {
protected:
  //! Id for this search
  static size_t _search_id;
  
  std::string _dbg;
  /*
   * Depth of the search (i.e., height of the search tree
   * visited so far.
   */
  size_t _depth;
  
  /**
   * Stores the number of backtracks during search.
   * A backtrack is a node for which all children have failed.
   */
  size_t _num_backtracks;
  
  /**
   * Stores the number of search nodes explored
   * during search.
   */
  size_t _num_nodes;
  
  /**
   * Stores the number of wrong decisions that have been made
   * during search. A wrong decision is represented by a leaf
   * of the search tree which has failed.
   */
  size_t _num_wrong_decisions;
  
  //! Specifies if debug option is on.
  bool _debug;
  
  //! Specifies if debug and trail debug options are on.
  bool _trail_debug;
  
  //! Specifies if the time-watcher is on
  bool _time_watcher;
  
  //! Specifies if the current search has been terminated.
  bool _search_out;
  
  //! Specifies if backtrack_out is active.
  bool _backtrack_out_on;
  
  //! Limit on the number of backtracks.
  size_t _backtracks_out;
  
  //! Specifies if nodes_out is active.
  bool _nodes_out_on;
  
  //! Limit on the number of nodes.
  size_t _nodes_out;
  
  //! Specifies if wrong_out is active.
  bool _wrong_out_on;
  
  //! Limit on the number of wrong decisions.
  size_t _wrong_out;
  
  //! Specifies if timeout_out is active.
  bool _timeout_out_on;
  
  //! Timeout value
  double _timeout_out;
  
  //! Reference to the constraint store to use during this search.
  ConstraintStorePtr _store;
  
  //! Reference to the current heuristic to use during search.
  HeuristicPtr _heuristic;
  
  //! Reference to the current backtrack manager.
  BacktrackManagerPtr _backtrack_manager;
  
  //! Solution manager
  SolutionManager* _solution_manager;
  
  /**
   * Initializes the current search (i.e., any parameter
   * used during search, as counters).
   */
  virtual void init_search ();
  
  /**
   * Tells whether the search has to be terminated due to
   * some limits (e.g., timeout, nodes_out, etc.).
   * @return true is the search has to be terminated,
   *         false otherwise.
   */
  virtual bool search_out ();
  
public:
  DepthFirstSearch ();
  
  virtual ~DepthFirstSearch ();
  
  /**
   * Set debug options.
   * @param debug_on boolean value indicating if debug
   *        should be enabled.
   * @note default debug is off.
   */
  void set_debug ( bool debug_on );
  
  /**
   * Set debug with trail option.
   * If enabled it prints debug and trail stack behaviours.
   * @param debug_on boolean value indicating if debug
   *        should be enabled.
   */
  void set_trail_debug ( bool debug_on );
  
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
  void set_solution_manager ( SolutionManager* sol_manager );
  
  /**
   * Sets a backtrackable manager to this class.
   * @param bkt_manager a reference to a backtrack manager.
   */
  void set_backtrack_manager ( BacktrackManagerPtr bkt_manager );
  
  /**
   * Returns the number of backtracks
   * performed by the search.
   * @return the number of backtracks.
   */
  size_t get_backtracks () const override;
  
  /**
   * Returns the number of nodes visited
   * by the search.
   * @return the number of visited nodes.
   */
  size_t get_nodes () const override;
  
  /**
   * Returns the number of wrong decisions made
   * during the search process.
   * @return the number of wrong decisions.
   * @note a decision is "wrong" depending on the search
   *       engine used to explore the search space.
   *       Usually, a wrong decision is represented by a
   *       leaf of the search tree which has failed.
   */
  size_t get_wrong_decisions () const override;
  
  /**
   * Set maximum number of solutions to be found.
   * @param num_sol the maximum number of solutions.
   * @note  -1 states for "find all solutions".
   */
  void set_solution_limit ( size_t num_sol ) override;
  
  /**
   * Imposes a timeoutlimit.
   * @param timeout timeout limit.
   * @note -1 for no timeout.
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
   * This function is called recursively.
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
   * Set a maximum number of backtracks to perform
   * during search.
   * @param the number of backtracks to consider as a limit
   *        during the search.
   */
  void set_backtrack_out ( size_t out_b ) override;
  
  /**
   * Set a maximum number of nodes to visit
   * during search.
   * @param the number of nodes to visit and to be considered
   *        as a limit during the search.
   */
  void set_nodes_out ( size_t out_n ) override;
  
  /**
   * Set a maximum number of wrong decisions to make
   * before exiting the search phase.
   * @param the number of wrong decisions to set
   *        as a limit during the search.
   */
  void set_wrong_decisions_out ( size_t out_w ) override;
  
  //! Print on standard output last solution found.
  void print_solution () const override;
  
  //! Print all solutions found so far
  void print_all_solutions () const override;
  
  /**
   * Print on standard output a solutions represented
   * by its index.
   * @param sol_idx the index of the solution to print.
   * @note first solution has index 1.
   */
  void print_solution ( size_t sol_idx ) const override;
  
  //! Prints info about the search engine
  void print () const override;

};

#endif /* defined(__NVIDIOSO__depth_first_search__) */
