//
//  int_lin_eq.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 20/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "int_lin_eq.h"

using namespace std;

IntLinEq::IntLinEq () :
FZNConstraint ( INT_LIN_EQ ) {
  
  // Set the event that trigger this constraint
  set_event( EventType::SINGLETON_EVT );
  set_event( EventType::MIN_EVT );
  set_event( EventType::MAX_EVT );
  set_event( EventType::BOUNDS_EVT );
}//IntLinEq

IntLinEq::IntLinEq ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
IntLinEq () {
  setup( vars, args );
}//IntLinEq

IntLinEq::~IntLinEq () {
  _as.clear();
  _bs.clear();
}//~IntLinEq

void
IntLinEq::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args ) {
  
  // Consistency checking in order to avoid more than one setup
  if ( _as.size() ) return;
  
  if ( vars.size() != (args.size() - 1) ) {
    throw NvdException ( "Bad parameters settings for int_lin_eq constraint." );
  }
  
  for ( int parameter = 0; parameter < args.size() -1; parameter++ )
    _as.push_back ( std::atoi( args[ parameter ].c_str() ) );
  
  for ( auto var : vars )
    _bs.push_back ( std::static_pointer_cast<IntVariable>( var ) );
  
  _c = std::atoi( args[ args.size() -1 ].c_str() );
}//setup

const std::vector<VariablePtr>
IntLinEq::scope () const {
  std::vector<VariablePtr> scope;
  for ( auto var : _bs )
    scope.push_back ( var );
  return scope;
}//scope

void
IntLinEq::consistency () {
  /**
   * This function propagates on bounds.
   * @see Apt K. Principles of constraint programming (CUP, 2003) pp 196.
   */
  if ( all_ground() ) return;
  
  // Positive coeff min/max (0/1), negative coeff min/max (2/3)
  int pos_neg [ 4 ];
  pos_neg[ 0 ] = pos_neg[ 1 ] =
  pos_neg[ 2 ] = pos_neg[ 3 ] = 0;
  
  for ( int i = 0; i < _bs.size(); i++ ) {
    if ( _as[ i ] >= 0 ) {
      pos_neg[ 0 ] += _as[ i ] * _bs[ i ]->min();
      pos_neg[ 1 ] += _as[ i ] * _bs[ i ]->max();
    }
    else {
      pos_neg[ 2 ] += -(_as[ i ] * _bs[ i ]->max());
      pos_neg[ 3 ] += -(_as[ i ] * _bs[ i ]->min());
    }
  }//i
  
  for ( int i = 0; i < _bs.size(); i++ ) {
    if ( _bs[ i ]->is_singleton() ) continue;
    int cur_lwb  = _bs[ i ]->min();
    int cur_upb  = _bs[ i ]->min();
    int v_coeff  = _as[ i ];
    
    if ( v_coeff > 0 ) {
      int alpha = (int) floor ( ( _c - (pos_neg[ 0 ] - (v_coeff * cur_lwb)) + pos_neg[ 2 ] ) / (v_coeff*1.0) );
      int gamma = (int)  ceil ( ( _c - (pos_neg[ 1 ] - (v_coeff * cur_upb)) + pos_neg[ 3 ] ) / (v_coeff*1.0) );
      
      // Update upper bound
      if ( alpha < cur_upb )
        _bs[ i ]->in_max( alpha );
      
      // Update lower bound
      if ( gamma > cur_lwb )
        _bs[ i ]->in_min( gamma );
    }
    else {
      int beta  = (int) ceil  ( (-_c + pos_neg[ 0 ] - (pos_neg[ 2 ] + (v_coeff * cur_upb))) / (-v_coeff*1.0) );
      int delta = (int) floor ( (-_c + pos_neg[ 1 ] - (pos_neg[ 3 ] + (v_coeff * cur_lwb))) / (-v_coeff*1.0) );
      
      // Update lower bound
      if ( beta > cur_lwb )
        _bs[ i ]->in_min( beta );
      
      // Update upper bound
      if ( delta > cur_upb )
        _bs[ i ]->in_max( delta );
    }
  }//i
}//consistency

bool
IntLinEq::satisfied () {
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
  
  return (product == _c);
}//satisfied

bool
IntLinEq::all_ground () {
  if (_bs.size() == 0 ) return false;
  for ( auto var : _bs )
    if ( !var->is_singleton() ) return false;
  return true;
}//all_ground

void
IntLinEq::print_semantic () const {
  FZNConstraint::print_semantic();
  std::cout << "Sum_{i \\in 1..n}: as[i].bs[i] = c ";
  std::cout << "where n is the common length of as and bs\n";
  std::cout << "int_lin_eq(array [int] of int: as, array [int] of var int: bs, int:c)\n";
  std::cout << "@note it is implicitly assumed that bs does not contain int values.\n";
}//print_semantic

