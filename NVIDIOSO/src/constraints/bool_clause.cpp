//
//  bool_clause.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "bool_clause.h"

BoolClause::BoolClause ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::BOOL_CLAUSE );
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//BoolClause

BoolClause::~BoolClause () {}

void
BoolClause::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
BoolClause::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
BoolClause::consistency ()
{
}//consistency

//! It checks if
bool
BoolClause::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
BoolClause::print_semantic () const
{
    BaseConstraint::print_semantic ();
}//print_semantic



