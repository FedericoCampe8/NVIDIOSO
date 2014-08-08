//
//  solver.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 27/06/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class provides the interface for a general constraint solver

#ifndef NVIDIOSO_solver_h
#define NVIDIOSO_solver_h

#include "cp_model.h"

class Solver {
public:
  virtual ~Solver() {};
  
  /**
   * Add a model to the solver.
   * @param model the (CP) model to add to the solver.
   * @note a solver can hold several models and decide
   *       both the model to run and the order in which
   *       run each model.
   */
  virtual void add_model ( CPModel* model ) = 0;
  
  /**
   * Removes a model actually destroying it.
   * @param the index of the model to destroy.
   */
  virtual void remove_model ( int model_idx ) = 0;
  
  /**
   * Returns a reference to model.
   * @param the index of the model to return.
   */
  virtual CPModel* get_model ( int model_idx ) const = 0;
  
  /**
   * It runs the solver in order to find 
   * a solution, the best solutions or other
   * solutions for all the models given to the solver.
   */
  virtual void run () = 0;
  
  /**
   * It runs the solver in order to find
   * a solution, the best solutions or other
   * solutions for the model specified by its index.
   * @param model_idx the index of the model to solve.
   */
  virtual void run ( int model_idx ) = 0;
  
  /**
   * Returns the number of models that are managed 
   * by this solver.
   * @return the number of models managed by this solver.
   */
  virtual int num_models () const = 0;
  
  /**
   * Returns the current number of runned models.
   * @return the number of models for which the run 
   *         function has been called.
   */
  virtual int num_solved_models () const = 0;
  
  /**
   * Returns the number of models for which a solution
   * has been found (out of the number of solved models).
   * @return the number of models for which a solution has
   *         been found.
   */
  virtual int sat_models () const = 0;
  
  /**
   * Returns the number of unsatisfiable models, i.e., 
   * the number of models with no solutions among those
   * that have been solved so far.
   * @return the number of unsatisfiable models.
   */
  virtual int unsat_models () const = 0;
  
  //! Print information about this solver.
  virtual void print () const = 0;
};


#endif
