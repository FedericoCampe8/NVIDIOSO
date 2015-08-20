//
//  int_min_c.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "int_min_c.h"

IntMinC::IntMinC ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::INT_MIN_C );
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntMinC

IntMinC::~IntMinC () {}

void
IntMinC::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
IntMinC::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
IntMinC::consistency ()
{
}//consistency

//! It checks if
bool
IntMinC::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
IntMinC::print_semantic () const
{
    BaseConstraint::print_semantic ();
}//print_semantic



