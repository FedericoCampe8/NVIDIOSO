//
//  set_ne.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "set_ne.h"

SetNe::SetNe () :
FZNConstraint ( SET_NE ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//SetNe

SetNe::SetNe ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
SetNe () {
  setup ( vars, args );
}//SetNe

SetNe::~SetNe () {}

void
SetNe::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
SetNe::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
SetNe::consistency ()
{
}//consistency

//! It checks if
bool
SetNe::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
SetNe::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



