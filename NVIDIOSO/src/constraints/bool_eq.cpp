//
//  bool_eq.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "bool_eq.h"

BoolEq::BoolEq () :
FZNConstraint ( BOOL_EQ ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntNe

BoolEq::BoolEq ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
BoolEq () {
  setup ( vars, args );
}//IntNe

BoolEq::~BoolEq () {}

void
BoolEq::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
BoolEq::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
BoolEq::consistency ()
{
}//consistency

//! It checks if
bool
BoolEq::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
BoolEq::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



