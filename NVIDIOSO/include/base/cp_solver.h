//
//  cp_solver.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 27/06/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class implements a solver for CP problems.
//  It has given a model as input and it explores the search space.
//

#ifndef NVIDIOSO_cp_solver_h
#define NVIDIOSO_cp_solver_h

#include "solver.h"

class CPSolver: public Solver {
protected:
  //! Debug info
  std::string _dbg;
  
  /**
   * CP models to be considered by this CPSolver.
   * The solver may decide which model to solve and in
   * which order solve it.
   */
  std::vector< CPModel * > _models;
  
  //! Number of solved models.
  int _solved_models;
  
  //! Number of models which have a solution
  int _sat_models;
  
  //! Number of unsatisfiable models.
  int _unsat_models;
  
  /**
   * It actually runs a CP Model.
   * @param a reference to a CP Model.
   */
  void run_model ( CPModel * model );
  
public:
  //! Constructor
  CPSolver ();
  
  /**
   * Constructor.
   * @param model a model to add to this CPSolver.
   */
  CPSolver ( CPModel* model );
  
  virtual ~CPSolver ();
  
  /**
   * Add a model to the solver.
   * @param model the reference to the (CP) model to add to the solver.
   * @note a solver can hold several models and decide
   *       both the model to run and the order in which
   *       run each model.
   */
  void add_model ( CPModel* model ) override;
  
  /**
   * Removes a model actually destroying it.
   * @param model_idx the index of the model to destroy.
   */
  void remove_model ( int model_idx ) override;
  
  /**
   * Returns a reference to model.
   * @param model_idx the index of the model to return.
   */
  CPModel* get_model ( int model_idx ) const override;
  
  /**
   * Further customizes a given model (identified by its index)
   * with user options.
   * @param i_data a reference to a input_data class where options
   *        are retrieved.
   * @param model_idx the index of the model to customize (default: 0,
   *        i.e., first model).
   */
  virtual void customize ( const InputData& i_data, int model_idx = 0 ) override;
  
  /**
   * It runs the solver in order to find
   * a solution, the best solutions or other
   * solutions w.r.t. the model given to
   * the solver.
   */
  void run();
  
  /**
   * It runs the solver in order to find
   * a solution, the best solutions or other
   * solutions for the model specified by its index.
   * @param model_idx the index of the model to solve.
   */
  void run ( int model_idx ) override;
  
  /**
   * Returns the number of models that are managed
   * by this solver.
   * @return the number of models managed by this solver.
   */
  int num_models () const override;
  
  /**
   * Returns the current number of runned models.
   * @return the number of models for which the run
   *         function has been called.
   */
  int num_solved_models () const override;
  
  /**
   * Returns the number of models for which a solution
   * has been found (out of the number of solved models).
   * @return the number of models for which a solution has
   *         been found.
   */
  int sat_models () const override;
  
  /**
   * Returns the number of unsatisfiable models, i.e.,
   * the number of models with no solutions among those
   * that have been solved so far.
   * @return the number of unsatisfiable models.
   */
  int unsat_models () const override;
  
  //! Print information about this solver.
  void print () const override;
};


#endif
