//
//  solution_manager.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/09/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class represents the interface of any solution manager
//  that can be attached to a search engine.
//

#ifndef __NVIDIOSO__solution_manager__
#define __NVIDIOSO__solution_manager__

#include "globals.h"
#include "variable.h"

class SolutionManager;
typedef std::unique_ptr<SolutionManager> SolutionManagerUPtr;
typedef std::shared_ptr<SolutionManager> SolutionManagerSPtr;

// Deprecated
typedef std::shared_ptr<SolutionManager> SolutionManagerPtr;
// Deprecated
typedef std::unique_ptr<SolutionManager> SolutionManagerUniquePtr;


class SolutionManager {
public:
  virtual ~SolutionManager() {};
  
  /**
   * Set the list of variables for which a solution
   * is required.
   * @param vars a vector of references to variables.
   */
  virtual void set_variables ( std::vector < Variable* >& vars ) = 0;
  
  /**
   * Prints the last solution found on standard output.
   * @note a solution is represented by the current values
   *       assigned to the variables attached to this solution manager.
   */
  virtual void print_solution () = 0;
  
  /**
   * Returns the number of solutions found so far.
   * @return the number of solutions.
   */
  virtual size_t number_of_solutions () = 0;
  
  /**
   * Get the last solution found.
   * @return a string representing the last solution found.
   */
  virtual std::string get_solution () const = 0;
  
  /**
   * Get the solution identified by its index.
   * @param sol_idx the index of the required solution.
   * @return a string representing the required solution.
   * @note first solution has index 1.
   */
  virtual std::string get_solution ( size_t sol_idx ) const = 0;
  
  /**
   * Get the all solutions found so far.
   * @return a vector of strings representing 
   *         all solutions found so far.
   */
  virtual std::vector< std::string > get_all_solutions () const = 0;
  
  /**
   * Sets a maximum number of solutions.
   * @param n_sol the number of solutions to compute.
   * @note -1 stands for "find all solutions".
   */
  virtual void set_solution_limit ( int n_sol ) = 0;
  
  /**
   * Increases the number of solutions found so far, computes the 
   * current solution, and it stores it.
   * States whether another solution is required by this solution manager in order 
   * to reach the total number of solutions.
   * @return true if no more solutions are required, false otherwise.
   */
  virtual bool notify () = 0;
  
  //! Print current variables' domains
  virtual void print_variables () = 0;
  
  //! Print information about this solution manager.
  virtual void print () const = 0;
};

#endif /* defined(__NVIDIOSO__solution_manager__) */
