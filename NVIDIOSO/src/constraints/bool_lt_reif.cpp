//
//  bool_lt_reif.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "bool_lt_reif.h"

BoolLtReif::BoolLtReif () :
FZNConstraint ( BOOL_LE_REIF ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntNe

BoolLtReif::BoolLtReif ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
BoolLtReif () {
  setup ( vars, args );
}//IntNe

BoolLtReif::~BoolLtReif () {}

void
BoolLtReif::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
BoolLtReif::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
BoolLtReif::consistency ()
{
}//consistency

//! It checks if
bool
BoolLtReif::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
BoolLtReif::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



