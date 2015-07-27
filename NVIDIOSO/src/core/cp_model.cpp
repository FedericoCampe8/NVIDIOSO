//
//  cp_model.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "cp_model.h"

using namespace std;

CPModel::CPModel () :
_search_engine ( nullptr ),
_store         ( nullptr ) {
  _model_id = glb_id_gen->get_int_id ();
}//CPModel

CPModel::~CPModel () {
}//~CPModel

int
CPModel::get_id () const {
  return _model_id;
}//get_id

size_t
CPModel::num_variables () const {
  return _variables.size ();
}//num_variables

size_t
CPModel::num_constraints () const {
  return _constraints.size ();
}//num_variables

void
CPModel::add_variable ( VariablePtr vpt ) {
  assert( vpt != nullptr );
  _variables.push_back( vpt );
}//add_variable

void
CPModel::add_constraint ( ConstraintPtr cpt ) {
  if( cpt == nullptr ) return;
  _constraints.push_back( cpt );
}//add_constraint

void
CPModel::add_search_engine ( SearchEnginePtr spt ) {
  assert( spt != nullptr );
  _search_engine = spt;
  
  if ( _store != nullptr ) {
    _search_engine->set_store( _store );
  }
}//add_search_engine

SearchEnginePtr
CPModel::get_search_engine () {
  return _search_engine;
}//get_search_engine

void
CPModel::add_constraint_store ( ConstraintStorePtr store ) {
  assert( store != nullptr );
  _store = store;
  
  if ( _search_engine != nullptr ) {
    _search_engine->set_store( _store );
  }
}//add_constraint_store

void
CPModel::init_constraint_store () {
  if ( _store == nullptr ) return;
  if ( _constraints.size () == 0 ) return;
  
  for ( auto c : _constraints ) {
    _store->impose( c );
  }
}//init_constraint_store

void
CPModel::finalize () {
}//finalize

void
CPModel::create_constraint_graph () {
  if ( !_constraints.size() ) return;
  for ( auto c : _constraints ) {
    c->attach_me_to_vars();
  }
}//create_constraint_graph

void
CPModel::attach_constraint_store () {
  if ( !_variables.size()) return;
  
  if ( _store == nullptr ) {
    throw NvdException("No constraint store to attach.");
  }
  
  for ( auto var : _variables ) {
    var->attach_store( _store );
  }
}//attach_constraint_store

void
CPModel::set_solutions_limit ( size_t sol_limit ) {
  _search_engine->set_solution_limit ( sol_limit );
}//set_solutions_limit

void
CPModel::set_timeout_limit ( double timeout ) {
  _search_engine->set_timeout_limit ( timeout );
}//set_timeout_limit

void
CPModel::print () const {
  cout << "CP Model:\n";
  cout << "|V|: " << _variables.size() << endl;
  cout << "|C|: " << _constraints.size() << endl;
  if ( _search_engine != nullptr ) {
    cout << "Search engine:\n";
    _search_engine->print();
  }
  for ( auto& v: _variables )
  {
  	v->print ();
  }
  for ( auto& c: _constraints )
  {
  	c->print ();
  }
}//print
