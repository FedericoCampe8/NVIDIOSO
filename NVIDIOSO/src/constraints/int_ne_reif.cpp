//
//  int_ne_reif.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "int_ne_reif.h"

IntNeReif::IntNeReif () :
FZNConstraint ( INT_NE_REIF ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntNeReif

IntNeReif::IntNeReif ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
IntNeReif () {
  setup ( vars, args );
}//IntNeReif

IntNeReif::~IntNeReif () {}

void
IntNeReif::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
IntNeReif::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
IntNeReif::consistency ()
{
}//consistency

//! It checks if
bool
IntNeReif::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
IntNeReif::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



