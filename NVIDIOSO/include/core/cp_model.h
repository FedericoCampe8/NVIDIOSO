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
//  @note: this class is also used to generate the constraint graph
//         when creating the objects.
//         For each variable, a link to the set of constraints is created.
//         For each constraint, a link to the variables in its scope is created.
//

#ifndef NVIDIOSO_cp_model_h
#define NVIDIOSO_cp_model_h

#include "globals.h"
#include "variable.h"
#include "constraint_store.h"
#include "search_engine.h"

class CPModel {
protected:
  
  //! Variables
  std::list < VariablePtr > _variables;
  //! Constraint Store
  ConstraintPtr _constraint_store;
  //! Search engine
  SearchEnginePtr _search_engine;
  
public:
  
  CPModel ();
  virtual ~CPModel();

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
   * @param ptr pointer to the search engine to use to
   *        explore the search space.
   */
  virtual void add_search_engine ( SearchEnginePtr ptr );
};

#endif
