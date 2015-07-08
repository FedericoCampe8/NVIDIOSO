//
//  set_intersect.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "set_intersect.h"

SetIntersect::SetIntersect () :
FZNConstraint ( SET_INTERSECT ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//SetIntersect

SetIntersect::SetIntersect ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
SetIntersect () {
  setup ( vars, args );
}//SetIntersect

SetIntersect::~SetIntersect () {}

void
SetIntersect::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
SetIntersect::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
SetIntersect::consistency ()
{
}//consistency

//! It checks if
bool
SetIntersect::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
SetIntersect::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



