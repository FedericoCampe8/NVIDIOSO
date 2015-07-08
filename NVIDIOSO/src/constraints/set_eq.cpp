//
//  set_eq.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "set_eq.h"

SetEq::SetEq () :
FZNConstraint ( SET_EQ ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//SetEq

SetEq::SetEq ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
SetEq () {
  setup ( vars, args );
}//SetEq

SetEq::~SetEq () {}

void
SetEq::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
SetEq::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
SetEq::consistency ()
{
}//consistency

//! It checks if
bool
SetEq::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
SetEq::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



