//
//  cp_model.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class represents a genral Constraint Programming model.
//  It holds the state concerning all the information needed to
//  explore the search space in order to find solutions.
//  It holds the following finite sets:
//    a) V of Variables, each associated with a finite domain D
//    b) C of constraints over the variables in V.
//       This set is represented by the correspondent Constraint Store.
//  This class represents general objects.
//  Specific implementations of the solver (i.e., using the CUDA framework)
//  should derive from this class and specilize the following methods.
//

#ifndef NVIDIOSO_cp_model_h
#define NVIDIOSO_cp_model_h

#include "globals.h"
#include "variable.h"
#include "constraint_store.h"
#include "search_engine.h"

class CPModel {
protected:
  //! Unique id for this model
  int _model_id;
  
  //! Variables
  std::vector < VariablePtr >    _variables;
  
  //! Constraint Store
  std::vector <  ConstraintPtr > _constraints;
  
  //! Search engine
  SearchEnginePtr _search_engine;
  
  //! Constraint store
  ConstraintStorePtr _store;
  
public:
  
  CPModel ();
  virtual ~CPModel();
  
  /**
   * Get the (unique) id of this model.
   * @return the model's id.
   */
  virtual int get_id () const;
  
  //! Return the current number of variabes in the model
  virtual size_t num_variables () const;
  
  //! Return the current number of constraints in the model
  virtual size_t num_constraints () const;
  
  /**
   * Add a variable to the model.
   * It linkes variables to constraints,
   * actually defining the constraint graph.
   * @param ptr pointer to the variable to add to the model
   */
  virtual void add_variable      ( VariablePtr ptr );
  
  /**
   * Add a constraint to the model.
   * It linkes constraints to variables, 
   * actually defining the constraint graph.
   * @param ptr pointer to the constraint to add to the model
   */
  virtual void add_constraint    ( ConstraintPtr ptr );
  
  /**
   * Add a search engine to the model.
   * @param ptr pointer to the search engine to use in order to
   *        explore the search space.
   * @note  if a constraint store is already present in the model,
   *        it sets the store into the given search engine.
   */
  virtual void add_search_engine ( SearchEnginePtr ptr );
  
  /**
   * Gets the search engine in order to run it.
   * @return a reference to the search engine in this model.
   */
  virtual SearchEnginePtr get_search_engine ();
  
  /**
   * Add a constraint store to the model.
   * @param store pointer to the constraint store to
   *        attach to the variables and propagate constraints.
   * @note this represents at least the first instance of constraint store.
   *       Every time this method is called, the variable's store will be
   *       updated with the given instance.
   * @note If a search engine is already present in the model, 
   *       it sets the given constraint store to the search engine.
   */
  virtual void add_constraint_store ( ConstraintStorePtr store );
  
  /**
   * Initializes the constraint store filling it with
   * the all the constraints into the model.
   */
  virtual void init_constraint_store ();
  
  /**
   * Finalizes the model.
   * @note This is an auxiliary method needed by some derived classes in
   *       order to finalize the model on different architectures.
   */
  virtual void finalize ();
  
  /**
   * Defines the constraint graphs actually attaching the constraints
   * to the variables.
   */
  virtual void create_constraint_graph ();
  
  /**
   * Sets the constraint store as current constraint store
   * for all the variables in the model.
   * When a variable changes its state, the constraint store
   * is automatically notified.
   */
  virtual void attach_constraint_store ();
  
  /**
   * Imposes a limit on the number of solutions.
   * @param sol_limit the maximum number of solutions for this model.
   * @note -1 means find all solutions.
   */
  virtual void set_solutions_limit ( size_t sol_limit );
  
  /**
   * Imposes a timeoutlimit.
   * @param timeout timeout limit.
   * @note -1 means no timeout.
   */
  virtual void set_timeout_limit ( double timeout );
  
  //! Print information about this CP Model.
  virtual void print () const;
};

#endif
