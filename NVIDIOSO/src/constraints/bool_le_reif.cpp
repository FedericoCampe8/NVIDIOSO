//
//  bool_le_reif.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "bool_le_reif.h"

BoolLeReif::BoolLeReif ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::BOOL_LE_REIF );
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//BoolLeReif

BoolLeReif::~BoolLeReif () {}

void
BoolLeReif::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
BoolLeReif::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
BoolLeReif::consistency ()
{
}//consistency

//! It checks if
bool
BoolLeReif::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
BoolLeReif::print_semantic () const
{
    BaseConstraint::print_semantic ();
}//print_semantic



