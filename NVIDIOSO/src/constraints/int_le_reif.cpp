//
//  int_le_reif.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "int_le_reif.h"

IntLeReif::IntLeReif () :
FZNConstraint ( INT_LE_REIF ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntLeReif

IntLeReif::IntLeReif ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
IntLeReif () {
  setup ( vars, args );
}//IntLeReif

IntLeReif::~IntLeReif () {}

void
IntLeReif::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
IntLeReif::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
IntLeReif::consistency ()
{
}//consistency

//! It checks if
bool
IntLeReif::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
IntLeReif::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



