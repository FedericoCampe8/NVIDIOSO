//
//  simple_solution_manager.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 09/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class implements a simple solutions manager.
//

#ifndef __NVIDIOSO__simple_solution_manager__
#define __NVIDIOSO__simple_solution_manager__

#include "solution_manager.h"

class SimpleSolutionManager : public SolutionManager {
protected:
  //! States wheter all solutions must be find or not.
  bool _find_all_solutions;
  
  /**
   * Stores the maximum  number of solutions
   * handled by this solution manager.
   * @note default value is 1;
   * @note if it is set to -1, all solutions are handled.
   */
  size_t _max_number_of_solutions;
  
  //! Stores the number of solutions found so far.
  size_t _number_of_solutions;
  
  /**
   * Stores the ordered list of variables
   * that represent a solution.
   * The order is given by variables' ids.
   */
  std::map < int, Variable * > _variables;
  
  /**
   * Store the string representations of the 
   * solutions found so far.
   */
  std::vector < std::string > _solution_strings;
  
public:
  //! Basic constructor
  SimpleSolutionManager ();
  
  /**
   * Constructor. It creates a new simple solution manager
   * attached to the given list of variables.
   * @param vars a vector of references to variables.
   */
  SimpleSolutionManager ( std::vector < Variable* >& vars );
  
  virtual ~SimpleSolutionManager();
  
  /**
   * Set the list of variables for which a solution
   * is required.
   * @param vars a vector of references to variables.
   */
  void set_variables ( std::vector < Variable* >& vars ) override;
  
  /**
   * Prints on standard output the last solution found.
   * @note a solution is represented by the current values
   *       assigned to the variables attached to this solution manager.
   */
  void print_solution () override;
  
  /**
   * Returns the number of solutions found so far.
   * @return the number of solutions.
   */
  size_t number_of_solutions () override;
  
  /**
   * Get the last solution found.
   * @return a string representing the last solution found.
   */
  std::string get_solution () const override;
  
  /**
   * Get the solution identified by its index.
   * @param sol_idx the index of the required solution.
   * @return a string representing the required solution.
   * @note first solution has index 1.
   */
  std::string get_solution ( size_t sol_idx ) const override;
  
  /**
   * Get the all solutions found so far.
   * @return a vector of strings representing
   *         all solutions found so far.
   */
  std::vector< std::string > get_all_solutions () const override;
  
  /**
   * Sets a maximum number of solutions.
   * @param n_sol the number of solutions to compute.
   * @note -1 stands for "find all solutions".
   */
  void set_solution_limit ( int n_sol ) override;
  
  /**
   * Increases the number of solutions found so far and
   * computes the current solution (also storing it).
   * States whether another solution is required by
   * this solution manager in order to reach the
   * total number of solutions.
   * @return true if no more solutions are required, false otherwise.
   */
  bool notify () override;
  
  //! Print current variables' domains
  void print_variables () override;
  
  //! Print information about this simple solution manager.
  void print () const override;
};

#endif /* defined(__NVIDIOSO__simple_solution_manager__) */
