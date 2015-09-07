//
//  cp_model.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
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
CPModel::get_id () const 
{
  return _model_id;
}//get_id

size_t
CPModel::num_variables () const 
{
  return _variables.size ();
}//num_variables

size_t
CPModel::num_constraints () const 
{
  return _constraints.size ();
}//num_variables

void
CPModel::add_aux_array ( std::string id, std::vector<int>& aux_info )
{
	_auxiliary_info [ id ] = aux_info;
}//add_variable

void
CPModel::add_variable ( VariablePtr vpt ) 
{
	assert( vpt != nullptr );
  	_variables.push_back( vpt );
}//add_variable

void
CPModel::add_constraint ( ConstraintPtr cpt ) 
{
	if( cpt == nullptr ) return;
	
	// Set shared arguments
	if ( _auxiliary_info.size () > 0 )
	{
		cpt->set_shared_arguments ( &_auxiliary_info );
	}
  	_constraints.push_back ( cpt );
}//add_constraint

void
CPModel::add_constraint ( GlobalConstraintPtr cpt ) 
{
	if( cpt == nullptr ) return;
	_glb_constraints.push_back ( cpt );
}//add_constraint

void
CPModel::add_search_engine ( SearchEnginePtr spt ) 
{
  assert( spt != nullptr );
  _search_engine = spt;
  
  if ( _store != nullptr ) 
  {
    _search_engine->set_store( _store );
  }
}//add_search_engine

SearchEnginePtr
CPModel::get_search_engine () 
{
  return _search_engine;
}//get_search_engine

void
CPModel::add_constraint_store ( ConstraintStorePtr store ) 
{
  assert( store != nullptr );
  _store = store;
  
  if ( _search_engine != nullptr ) 
  {
    _search_engine->set_store( _store );
  }
}//add_constraint_store

void
CPModel::init_constraint_store () 
{
  if ( _store == nullptr ) return;
  if ( _constraints.size () == 0 ) return;
  
  // Add base constraints
  for ( auto c : _constraints ) 
  {
	 _store->impose( c );
  }
  
  // Add global constraints
  for ( auto c : _glb_constraints ) 
  {
	 _store->impose( c );
  }
}//init_constraint_store

void
CPModel::finalize () 
{
}//finalize

void
CPModel::create_constraint_graph () 
{
  if ( !_constraints.size() ) return;
  
  // Attach base constraints to variables
  for ( auto c : _constraints ) 
  {
	 c->attach_me_to_vars();
  }
  
  // Attach global constraints to variable
  for ( auto c : _glb_constraints ) 
  {
	 c->attach_me_to_vars();
  }
}//create_constraint_graph

void
CPModel::attach_constraint_store () 
{
  if ( !_variables.size()) return;
  
  if ( _store == nullptr ) {
    throw NvdException("No constraint store to attach.");
  }
  
  for ( auto var : _variables ) {
    var->attach_store( _store );
  }
}//attach_constraint_store

void
CPModel::set_solutions_limit ( size_t sol_limit ) 
{
	_search_engine->set_solution_limit ( sol_limit );
}//set_solutions_limit

void
CPModel::set_timeout_limit ( double timeout ) 
{
	_search_engine->set_timeout_limit ( timeout );
}//set_timeout_limit

void
CPModel::print () const 
{
  cout << "CP Model:\n";
  cout << "|V|:  " << _variables.size()       << endl;
  cout << "|C|:  " << _constraints.size()     << endl;
  cout << "|GC|: " << _glb_constraints.size() << endl;
  if ( _search_engine != nullptr ) 
  {
    cout << "Search engine:\n";
    _search_engine->print();
  }
  
  cout << "=== AUXILIARY ARRAYS ===\n";
  for ( auto& a: _auxiliary_info )
  {
  	cout << "ID: " << a.first << ":\n";
  	cout << "[ -, ";
  	for ( auto& elem : a.second )
  	{
  		cout << elem << ", ";
  	}
  	cout << "- ]" << endl;
  }
  cout << "=================\n";
  cout << "=== VARIABLES ===\n";
  for ( auto& v: _variables )
  {
  	v->print ();
  }
  cout << "=================\n";
  cout << "=== CONSTRAINTS ===\n";
  for ( auto& c: _constraints )
  {
  	c->print ();
  }
  cout << "=================\n";
  cout << "=== GLOBAL CONSTRAINTS ===\n";
  for ( auto& c: _glb_constraints )
  {
  	c->print ();
  }
  cout << "=================\n";
}//print
