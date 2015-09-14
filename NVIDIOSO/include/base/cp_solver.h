//
//  cp_solver.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 27/06/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
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
   * Hash table of CP models to be solved by the (CP) solver.
   * The table contains <key, value> pairs where:
   * - key: model (unique) identified;
   * - value: (unique) pointer to the (CP) model.
   */
  std::unordered_map< int, std::unique_ptr<CPModel> > _models;
  
  //! Number of solved models.
  std::size_t _solved_models;
  
  //! Number of models which have a solution
  std::size_t _sat_models;
  
  //! Number of unsatisfiable models.
  std::size_t _unsat_models;
  
  /**
   * It actually runs a CP Model.
   * @param a reference to a CP Model.
   */
  void run_model ( CPModel * model );
  
public:
  //! Constructor.
  CPSolver ();
  
  /**
   * Constructor.
   * @param model a model to add to this CPSolver.
   */
  CPSolver ( CPModelUPtr model );
  
  /**
   * Copy constructor.
   * @note this means that it is actually possible to copy a solver.
   *       Copying a solver means that all its internal models are copied which 
   *       may be expensive.
   */
  CPSolver ( const CPSolver& other );
  
  /**
   * Assignment operator.
   * @note this means that it is actually possible to copy a solver.
   *       Copying a solver means that all its internal models are copied which 
   *       may be expensive.
   */
  CPSolver& operator= ( const CPSolver& other );
  
  //! Destructor.
  virtual ~CPSolver ();
  
  /**
   * Add a model to the solver.
   * @param model the reference to the (CP) model to add to the solver.
   * @note a solver can hold several models and decide
   *       both the model to run and the order in which
   *       run each model.
   */
  void add_model ( CPModelUPtr model ) override;
  
  /**
   * Removes a model destroying it.
   * @param model_id the (unique) identifier of the model to destroy.
   * @note does nothing if model is not present.
   */
  void remove_model ( int model_id ) override;
  
  /**
   * Returns a reference to a selected model.
   * @param model_id the index of the model to return.
   * @note returns NULL if model is not present.
   */
   const CPModel * get_model ( int model_id ) const override;
  
  /**
   * --- THIS FUNCTION IS DEPRECATED ---
   * Customizes a given model (identified by its index) with user options.
   * @param i_data a reference to a input_data class where options are retrieved.
   * @param model_idx the index of the model to customize (default: 0, i.e., first model).
   * @note in the current implementation (1.0) this function sets timout limit, 
   *       solution limit and time watcher.
   * @note does nothing if model is not present.
   */
  virtual void customize ( const InputData& i_data, int model_id ) override;
  
  //! Runs the solver on all the models.
  void run ();
  
  /**
   * It runs the solver in order to find a solution, the best solutions or other
   * solutions for the model specified by its index.
   * @param model_id the (unique) identifier of the model to solve.
   * @note does nothing if model is not present.
   */
  void run ( int model_id ) override;
  
  /**
   * Returns the number of models that are managed by this solver.
   * @return the number of models managed by this solver.
   */
  std::size_t num_models () const override;
  
  /**
   * Returns the current number of run models.
   * @return the number of models for which the run function has been called.
   */
  std::size_t num_solved_models () const override;
  
  /**
   * Returns the number of models for which a solution
   * has been found (out of the number of solved models).
   * @return the number of models for which a solution has been found.
   */
  std::size_t sat_models () const override;
  
  /**
   * Returns the number of unsatisfiable models, i.e., 
   * the number of models with no solutions among those that have been solved so far.
   * @return the number of unsatisfiable models.
   */
  std::size_t unsat_models () const override;
  
  //! Print information about this solver.
  void print () const override;
};


#endif
