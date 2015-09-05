//
//  local_search_solution_manager.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/09/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements and extends a simple solution manager to be attached
//  and used by Local Search strategies.
//  It handles values on objective function, epsilons, and models where the objective 
//  is represented only by constraints and their satisfiability.
//

#ifndef __NVIDIOSO__local_search_solution_manager__
#define __NVIDIOSO__local_search_solution_manager__

#include "simple_solution_manager.h"

class LocalSearchSolutionManager : public SimpleSolutionManager {
protected:
	
	//! Epsilon on the objective value to achieve before terminating search
	double _objective_epsilon;
	
	/**
	 * Current best objective value.
	 * @note this is not always represented by the value hold by the objective variable.
	 *       It can be also represented by the value of unsatisfied constraints.
	 */ 
	double _objective_val;
	
	//! States if only constraint satisfiability is considered as objective
	bool _objective_sat;
	
	//! Number of unsatisfied constraints acceptable as a solution
	std::size_t _unsat_constraints_out; 
	
	//! Pointer to the variable representing the objective value
	Variable * _objective_var;
 
	//! True if epsilon value has been reached, false otherwise
	bool epsilon_satisfied ();
	
	//! True if the number of satisfied constraints has been reached, false otherwise
	bool constraint_satisfied ( std::size_t sat_con ); 
	
public:
  //! Basic constructor
  LocalSearchSolutionManager ();
   
  /**
   * Constructor. It creates a new simple solution manager
   * attached to the given list of variables.
   * @param vars a vector of references to variables.
   * @param pointer to the objective variable if any. 
   */
  LocalSearchSolutionManager ( std::vector < Variable* >& vars, Variable* obj_var=nullptr );
  
  virtual ~LocalSearchSolutionManager();
     
  /**
   * Set the objective variable.
   * @param pointer to the objective variable.
   */
  void set_obj_variable ( Variable* obj_var );
   
  /**
   * Set epsilon on the objective value.
   * @param n_sol the number of solutions to compute.
   * @note -1 stands for "find all solutions".
   */
  virtual void set_epsilon_limit ( double epsilon );
  
  /**
   * Set satisfiability as objective function.
   * @param sat set/unset satisfiability as objective, default true.
   */
  virtual void use_satisfiability_obj ( bool sat = true );
   
  /**
   * Notify the manager about a value which is changed due to propagation.
   * This function is needed to record solutions found during local search which 
   * may be worsened by future labelings. 
   */
  virtual bool notify_on_propagation ( std::size_t value );
  
  //! Print information about this simple solution manager.
  void print () const override;
};

#endif /* defined(__NVIDIOSO__local_search_solution_manager__) */
