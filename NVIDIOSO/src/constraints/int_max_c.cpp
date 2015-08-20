//
//  int_max_c.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "int_max_c.h"

IntMaxC::IntMaxC ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::INT_MAX_C );
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntMaxC

IntMaxC::~IntMaxC () {}

void
IntMaxC::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
IntMaxC::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
IntMaxC::consistency ()
{
}//consistency

//! It checks if
bool
IntMaxC::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
IntMaxC::print_semantic () const
{
    BaseConstraint::print_semantic ();
}//print_semantic



