//
//  int_mod.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "int_mod.h"

IntMod::IntMod () :
FZNConstraint ( INT_MOD ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntMod

IntMod::IntMod ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
IntMod () {
  setup ( vars, args );
}//IntMod

IntMod::~IntMod () {}

void
IntMod::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
IntMod::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
IntMod::consistency ()
{
}//consistency

//! It checks if
bool
IntMod::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
IntMod::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



