//
//  cp_solver.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "cp_solver.h"

using namespace std;

CPSolver::CPSolver () :
_dbg           ( "CPSolver - " ),
_solved_models ( 0 ),
_sat_models    ( 0 ),
_unsat_models  ( 0 ) {
}//CPSolver

CPSolver::CPSolver ( CPModel* model ) :
CPSolver () {
  if ( model != nullptr ) add_model ( model );
}//CPSolver

CPSolver::~CPSolver () {
  if ( num_models() > 0 ) {
    for ( auto model : _models )
      delete model;
    _models.clear();
  }
}//~CPSolver

void
CPSolver::add_model ( CPModel* model ) {
  if ( model != nullptr ) {
  
    if ( !_models.size() ) {
      _models.push_back ( model );
      return;
    }
    
    int model_id = model->get_id();
    for ( auto model : _models )
      if ( model_id == model->get_id() ) return;
    
    _models.push_back ( model );
  }
}//add_model

void
CPSolver::remove_model ( int model_idx ) {
  if ( model_idx >= 0 && model_idx < _models.size () ) {
    _models.erase ( _models.begin() + model_idx );
  }
}//remove_model

CPModel*
CPSolver::get_model ( int model_idx ) const {
  if ( model_idx >= 0 && model_idx < _models.size () ) {
    return _models[ model_idx ];
  }
  return nullptr;
}//get_model

void
CPSolver::customize ( const InputData& i_data, int model_idx ) {
  if ( _models.size () == 0 ) return;
  if ( model_idx >= 0 && model_idx < _models.size () ) {
    _models[ model_idx ]->set_timeout_limit  ( i_data.timeout   () );
    _models[ model_idx ]->set_solutions_limit( i_data.max_n_sol () );
    (_models[ model_idx ]->get_search_engine())->set_time_watcher( i_data.timer() );
  }
}//customize

void
CPSolver::run_model ( CPModel * model ) {
  /*
   * Create the constraint graph, i.e., 
   * attach constraints to FD variables.
   */
  try {
    model->create_constraint_graph ();
  } catch ( NvdException& e ) {
    e.what();
    _solved_models++;
    return;
  }
  
  /*
   * Fill constraint store with all the constraints
   * in the model for re-evaluation.
   */
  model->init_constraint_store ();
  
  /*
   * Attach the constraint store to each variable
   * in the model. In this way, any time a variable
   * changes its state, it will automatically inform
   * the constraint store.
   */
  try 
  {
    model->attach_constraint_store ();
  } catch ( NvdException& e ) {
    e.what();
    _solved_models++;
    return;
  }

  // Run the search engine of this model.
  bool solution = false;
  if ( model->get_search_engine() != nullptr ) {
    try {
      solution = (model->get_search_engine())->labeling();
    } catch ( NvdException& e ) {
      e.what();
      std::cerr << "Model_" << model->get_id() <<
      " terminated improperly:\n";
      std::cerr << e.what() << std::endl;
    }
    
    if ( solution ) _sat_models++;
    else            _unsat_models++;
  }
  else {
    ostringstream s;
    s << model->get_id();
    LogMsg << _dbg << "No Search Engine for model_" << s.str() << endl;
  }
  
  // Increase the number of solved models
  _solved_models++;
}//run_model

void
CPSolver::run() {
  if ( _models.size () == 0) return;
  for ( int i = 0; i < _models.size(); i++ ) {
    run ( i );
  }//i
}//run

void
CPSolver::run ( int model_idx ) {
  if ( _models.size () == 0 ) return;
  if ( model_idx >= 0 && model_idx < _models.size () ) {
    run_model ( _models[ model_idx ] );
  }
}//run

int
CPSolver::num_models () const {
  return (int) _models.size();
}//num_models

int
CPSolver::num_solved_models () const {
  return _solved_models;
}//num_solved_models

int
CPSolver::sat_models () const {
  return _sat_models;
}//sat_models

int
CPSolver::unsat_models () const {
  return _unsat_models;
}//unsat_models

void
CPSolver::print () const {
  cout << "CPSolver:\n";
  cout << "Total models:            " << num_models () << endl;
  cout << "Solved models:           " << num_solved_models () << endl;
  cout << "Models with solution:    " << sat_models () << endl;
  cout << "Models without solution: " << unsat_models () << endl;
}//print





