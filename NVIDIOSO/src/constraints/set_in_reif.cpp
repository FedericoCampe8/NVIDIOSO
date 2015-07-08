//
//  set_in_reif.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "set_in_reif.h"

SetInReif::SetInReif () :
FZNConstraint ( SET_IN_REIF ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//SetInReif

SetInReif::SetInReif ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
SetInReif () {
  setup ( vars, args );
}//SetInReif

SetInReif::~SetInReif () {}

void
SetInReif::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
SetInReif::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
SetInReif::consistency ()
{
}//consistency

//! It checks if
bool
SetInReif::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
SetInReif::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



