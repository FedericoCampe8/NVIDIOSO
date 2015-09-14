//
//  cp_solver.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/08/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "cp_solver.h"

using std::cout;
using std::endl;
using std::string;

CPSolver::CPSolver () :
  _dbg           ( "CPSolver - " ),
  _solved_models ( 0 ),
  _sat_models    ( 0 ),
  _unsat_models  ( 0 ) {
}//CPSolver

CPSolver::CPSolver ( CPModelUPtr model ) :
  CPSolver () {
  if ( model != nullptr )
  {
    add_model ( std::move ( model ) );
  } 
  else
  {
    LogMsgW << _dbg << "[CPSolver] Model created with a NULL pointer model" << endl;
  }
}//CPSolver

CPSolver::CPSolver ( const CPSolver& other ) {
  _dbg = "CPSolver - ";
  _solved_models = other._solved_models;
  _sat_models    = other._sat_models;
  _unsat_models  = other._unsat_models;
  
  for ( auto& mdl : other._models )
  {
    _models [ mdl.first ] = std::unique_ptr<CPModel> ( new CPModel ( *(mdl.second.get ()) ) );
  }
}//CPSolver

CPSolver&
CPSolver::operator= ( const CPSolver& other ) {
  if ( this != &other )
  {
    _dbg = "CPSolver - ";
    _solved_models = other._solved_models;
    _sat_models    = other._sat_models;
    _unsat_models  = other._unsat_models;
    
    _models.clear ();
    for ( auto& mdl : other._models )
    {
      _models [ mdl.first ] = std::unique_ptr<CPModel> ( new CPModel ( *(mdl.second.get ()) ) );
    }
  }
  
  return *this;
}//CPSolver

CPSolver::~CPSolver () {
  _models.clear();
}//~CPSolver

void
CPSolver::add_model ( CPModelUPtr model ) 
{
  // Sanity check
  if ( model != nullptr ) 
  {// If no model is present add it and return
    if ( !_models.size() ) 
    {
      try
      {
        _models [ model->get_id() ] = std::move ( model ); 
      }
      catch ( std::exception& e )
      {
        LogMsgE << _dbg << "[add_mode] Cannot add the model to the Solver" << endl;
      }
      
      return;
    }
    
    // If models are already present: check to avoid duplicates
    int model_id = model->get_id();
    auto it = _models.find ( model_id );
    if ( it == _models.end() )
    {
      try
      {
        _models [ model->get_id() ] = std::move ( model ); 
      }
      catch ( std::exception& e )
      {
        LogMsgE << _dbg << "[add_mode] Cannot add the model to the Solver" << endl;
      }
    }
  }
}//add_model

void
CPSolver::remove_model ( int model_id ) 
{
  auto it = _models.find ( model_id );
  if ( it != _models.end() )
  {
    _models.erase ( it );
  }
}//remove_model

const CPModel*
CPSolver::get_model ( int model_id ) const 
{
  auto it = _models.find ( model_id );
  if ( it != _models.end() )
  {
    return _models.at( model_id ).get ();
  }
  return nullptr;
}//get_model

void
CPSolver::customize ( const InputData& i_data, int model_id ) 
{
  auto it = _models.find ( model_id );
  if ( it != _models.end() )
  {
    _models[ model_id ]->set_timeout_limit  ( i_data.timeout   () );
    _models[ model_id ]->set_solutions_limit( i_data.max_n_sol () );
    if ( _models[ model_id ]->get_search_engine() != nullptr ) 
    {
      (_models[ model_id ]->get_search_engine())->set_time_watcher( i_data.timer() );
    }
  }
}//customize

void
CPSolver::run_model ( CPModel * model ) 
{
  // Create the constraint graph, i.e., attach constraints to FD variables.
  try 
  {
    model->create_constraint_graph ();
  } 
  catch ( NvdException& e ) 
  {
    e.what();
    _solved_models++;
    return;
  }
  
  // Fill constraint store with all the constraints in the model for re-evaluation.
  model->init_constraint_store ();
  
  /*
   * Attach the constraint store to each variable in the model. 
   * Any time a variable changes its state, it will automatically inform the constraint store.
   * @note see "Observer" pattern.
   */
  try 
  {
    model->attach_constraint_store ();
  } 
  catch ( NvdException& e ) 
  {
    e.what();
    _solved_models++;
    return;
  }

  // Run the search engine of this model.
  bool solution = false;
  if ( model->get_search_engine() != nullptr ) 
  {
    try 
    {
      solution = (model->get_search_engine())->labeling();
    } 
    catch ( NvdException& e ) 
    {
      LogMsgE << "Model_" << model->get_id() << " terminated improperly: " <<
      std::string ( e.what () ) << std::endl;
    }
    
    if ( solution )
    {
      _sat_models++;
    } 
    else
    {
      _unsat_models++;
    }            
  }
  else 
  {
    std::ostringstream s;
    s << model->get_id();
    LogMsgW << _dbg << "No Search Engine for model_" << s.str() << endl;
  }
  
  // Increase the number of solved models
  _solved_models++;
}//run_model

void
CPSolver::run () 
{
  for ( auto& mdl : _models )
  {
    run_model ( mdl.second.get() );
  }
}//run

void
CPSolver::run ( int model_id ) 
{
  auto it = _models.find ( model_id );
  if ( it != _models.end () )
  {
    run_model ( _models[ model_id ].get() );
  }
}//run

std::size_t
CPSolver::num_models () const 
{
  return _models.size();
}//num_models

std::size_t
CPSolver::num_solved_models () const 
{
  return _solved_models;
}//num_solved_models

std::size_t
CPSolver::sat_models () const 
{
  return _sat_models;
}//sat_models

std::size_t
CPSolver::unsat_models () const 
{
  return _unsat_models;
}//unsat_models

void
CPSolver::print () const 
{
  cout << "CPSolver:\n";
  cout << "Number of attached models: " << num_models () << endl;
  cout << "Solved models:             " << num_solved_models () << endl;
  cout << "Models with solution:      " << sat_models () << endl;
  cout << "Models without solution:   " << unsat_models () << endl;
}//print
