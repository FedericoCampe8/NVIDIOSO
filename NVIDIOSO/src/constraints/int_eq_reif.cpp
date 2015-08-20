//
//  int_eq_reif.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "int_eq_reif.h"

IntEqReif::IntEqReif ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::INT_EQ_REIF );
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntEqReif

IntEqReif::~IntEqReif () {}

void
IntEqReif::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
IntEqReif::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
IntEqReif::consistency ()
{
}//consistency

//! It checks if
bool
IntEqReif::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
IntEqReif::print_semantic () const
{
    BaseConstraint::print_semantic ();
}//print_semantic



