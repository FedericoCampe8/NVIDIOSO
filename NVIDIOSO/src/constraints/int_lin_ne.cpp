//
//  int_lin_ne.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 11/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "int_lin_ne.h"

using namespace std;

IntLinNe::IntLinNe () :
FZNConstraint ( INT_LIN_NE ) {
  
  // Set the event that trigger this constraint
  set_event( EventType::SINGLETON_EVT );
}//IntLinNe

IntLinNe::IntLinNe ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
IntLinNe () {
  setup( vars, args );
}//IntLinNe

IntLinNe::~IntLinNe () {
  _as.clear();
  _bs.clear();
}//~IntLinNe

void
IntLinNe::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args ) {
  
  // Consistency checking in order to avoid more than one setup
  if ( _as.size() ) return;
  
  if ( vars.size () != (args.size() - 1) ) 
  {
    throw NvdException ( "Bad parameters settings for int_lin_ne constraint." );
  }
  
  for ( int parameter = 0; parameter < args.size() -1; parameter++ )
  {
    _as.push_back ( std::atoi( args[ parameter ].c_str() ) );
    _arguments.push_back ( _as.back() );
  }
  
  for ( auto var : vars )
    _bs.push_back ( std::static_pointer_cast<IntVariable>( var ) );
  
  _c = std::atoi( args[ args.size() -1 ].c_str() );
  _arguments.push_back ( _c );
}//setup

const std::vector<VariablePtr>
IntLinNe::scope () const {
  std::vector<VariablePtr> scope;
  for ( auto var : _bs )
    scope.push_back ( var );
  return scope;
}//scope

void
IntLinNe::consistency () {
  /**
   * This function propagates only when there is just
   * variables that is not still assigned.
   * Otherwise it returns without any check.
   */
  if ( all_ground() )       return;
  if ( !only_one_not_ground() ) return;
  
  int product = 0;
  int non_ground_idx = -1;
  for ( int idx = 0; idx < _bs.size(); idx++ ) {
    if ( !_bs[ idx ]->is_singleton() ) {
      non_ground_idx = idx;
      continue;
    }
    product += _as[ idx ] * _bs[ idx ]->min();
  }//var
  
  // a + kx != c -> x != (c - a) / k
  int avoid = (_c - product) / _as[ non_ground_idx ];
  if ( non_ground_idx != -1 )
    _bs[ non_ground_idx ]->subtract( avoid );
}//consistency

bool
IntLinNe::satisfied () {
  /*
   * If not variables are ground, then there
   * is not enough information to state whether the constraint
   * is satisfied or not.
   * Return true.
   */
  if ( !all_ground() ) return true;
  
  int product = 0;
  for ( int idx = 0; idx < _bs.size(); idx++ )
    product += _as[ idx ] * _bs[ idx ]->min();
  
  return (product != _c);
}//satisfied

bool
IntLinNe::all_ground () {
  if (_bs.size() == 0 ) return false;
  for ( auto var : _bs )
    if ( !var->is_singleton() ) return false;
  return true;
}//all_ground

bool
IntLinNe::at_least_one_ground () {
  if (_bs.size() == 0 ) return false;
  for ( auto var : _bs )
    if ( var->is_singleton() ) return true;
  return false;
}//all_ground

bool
IntLinNe::only_one_not_ground () {
  if (_bs.size() == 0 ) return false;
  int num_ground = 0;
  for ( auto var : _bs ) {
    if ( var->is_singleton() ) num_ground++;
  }
  if ( num_ground == (_bs.size() - 1) ) return true;
  return false;
}//all_ground

void
IntLinNe::print_semantic () const {
  FZNConstraint::print_semantic();
  std::cout << "Sum_{i \\in 1..n}: as[i].bs[i] != c ";
  std::cout << "where n is the common length of as and bs\n";
  std::cout << "int_lin_ne(array [int] of int: as, array [int] of var int: bs, int:c)\n";
  std::cout << "@note it is implicitly assumed that bs does not contain int values.\n";
}//print_semantic




