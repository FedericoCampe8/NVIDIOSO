//
//  set_diff.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "set_diff.h"

SetDiff::SetDiff () :
FZNConstraint ( SET_DIFF ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//SetDiff

SetDiff::SetDiff ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
SetDiff () {
  setup ( vars, args );
}//SetDiff

SetDiff::~SetDiff () {}

void
SetDiff::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
SetDiff::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
SetDiff::consistency ()
{
}//consistency

//! It checks if
bool
SetDiff::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
SetDiff::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



