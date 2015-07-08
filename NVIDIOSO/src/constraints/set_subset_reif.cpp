//
//  set_subset_reif.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "set_subset_reif.h"

SetSubsetReif::SetSubsetReif () :
FZNConstraint ( SET_SUBSET_REIF ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//SetSubsetReif

SetSubsetReif::SetSubsetReif ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
SetSubsetReif () {
  setup ( vars, args );
}//SetSubsetReif

SetSubsetReif::~SetSubsetReif () {}

void
SetSubsetReif::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
SetSubsetReif::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
SetSubsetReif::consistency ()
{
}//consistency

//! It checks if
bool
SetSubsetReif::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
SetSubsetReif::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



