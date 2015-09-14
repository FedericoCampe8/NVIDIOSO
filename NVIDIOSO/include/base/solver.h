//
//  solver.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 27/06/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class provides the interface for a general constraint solver.
//

#ifndef NVIDIOSO_solver_h
#define NVIDIOSO_solver_h

#include <iostream>

#include "cp_model.h"
#include "input_data.h"

class Solver {
public:
  virtual ~Solver() {};
  
  /**
   * Add a model to the solver.
   * @param model the reference to the (CP) model to add to the solver.
   * @note a solver can hold several models and decide
   *       both the model to run and the order in which
   *       run each model.
   */
  virtual void add_model ( CPModelUPtr model ) = 0;
  
  /**
   * Removes a model destroying it.
   * @param model_id the (unique) identifier of the model to destroy.
   * @note does nothing if model is not present.
   */
  virtual void remove_model ( int model_id ) = 0;
  
  /**
   * Returns a reference to a selected model.
   * @param model_id the index of the model to return.
   * @note returns unique pointer instantiated at NULL if model is not present.
   * @note ownership of the object still remains to the solver, which means that
   *       the model will be destroyed when the destuctor of this class will be invoked
   */
  virtual const CPModel * get_model ( int model_id ) const = 0;
  
  /**
   * --- THIS FUNCTION IS DEPRECATED ---
   * Customizes a given model (identified by its index) with user options.
   * @param i_data a reference to a input_data class where options are retrieved.
   * @param model_idx the index of the model to customize (default: 0, i.e., first model).
   * @note does nothing if model is not present.
   */
  virtual void customize ( const InputData& i_data, int model_id ) = 0;
  
  /**
   * It runs the solver in order to find  a solution, the best solutions 
   * or all solutions for all the models given to the solver.
   */
  virtual void run () = 0;
  
  /**
   * It runs the solver in order to find a solution, the best solutions or other
   * solutions for the model specified by its index.
   * @param model_id the (unique) identifier of the model to solve.
   * @note does nothing if model is not present.
   */
  virtual void run ( int model_id ) = 0;
  
  /**
   * Returns the number of models that are managed by this solver.
   * @return the number of models managed by this solver.
   */
  virtual std::size_t num_models () const = 0;
  
  /**
   * Returns the current number of run models.
   * @return the number of models for which the run function has been called.
   */
  virtual std::size_t num_solved_models () const = 0;
  
  /**
   * Returns the number of models for which a solution
   * has been found (out of the number of solved models).
   * @return the number of models for which a solution has been found.
   */
  virtual std::size_t sat_models () const = 0;
  
  /**
   * Returns the number of unsatisfiable models, i.e., 
   * the number of models with no solutions among those that have been solved so far.
   * @return the number of unsatisfiable models.
   */
  virtual std::size_t unsat_models () const = 0;
  
  //! Print information about this solver.
  virtual void print () const = 0;
};


#endif
