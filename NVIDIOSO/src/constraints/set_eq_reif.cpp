//
//  set_eq_reif.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "set_eq_reif.h"

SetEqReif::SetEqReif () :
FZNConstraint ( SET_EQ_REIF ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//SetEqReif

SetEqReif::SetEqReif ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
SetEqReif () {
  setup ( vars, args );
}//SetEqReif

SetEqReif::~SetEqReif () {}

void
SetEqReif::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
SetEqReif::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
SetEqReif::consistency ()
{
}//consistency

//! It checks if
bool
SetEqReif::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
SetEqReif::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



