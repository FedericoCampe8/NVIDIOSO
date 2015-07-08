//
//  int_plus.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "int_plus.h"

IntPlus::IntPlus () :
FZNConstraint ( INT_PLUS ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntPlus

IntPlus::IntPlus ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
IntPlus () {
  setup ( vars, args );
}//IntPlus

IntPlus::~IntPlus () {}

void
IntPlus::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
IntPlus::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
IntPlus::consistency ()
{
}//consistency

//! It checks if
bool
IntPlus::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
IntPlus::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



