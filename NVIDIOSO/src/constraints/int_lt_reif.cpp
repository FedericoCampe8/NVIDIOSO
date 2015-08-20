//
//  int_lt_reif.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "int_lt_reif.h"

IntLtReif::IntLtReif ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::INT_LT_REIF );
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntLtReif

IntLtReif::~IntLtReif () {}

void
IntLtReif::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
IntLtReif::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
IntLtReif::consistency ()
{
}//consistency

//! It checks if
bool
IntLtReif::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
IntLtReif::print_semantic () const
{
    BaseConstraint::print_semantic ();
}//print_semantic



