//
//  int_lin_eq_reif.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "int_lin_eq_reif.h"

IntLinEqReif::IntLinEqReif () :
FZNConstraint ( INT_LIN_EQ_REIF ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntLinEqReif

IntLinEqReif::IntLinEqReif ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
IntLinEqReif () {
  setup ( vars, args );
}//IntLinEqReif

IntLinEqReif::~IntLinEqReif () {}

void
IntLinEqReif::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
IntLinEqReif::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
IntLinEqReif::consistency ()
{
}//consistency

//! It checks if
bool
IntLinEqReif::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
IntLinEqReif::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



