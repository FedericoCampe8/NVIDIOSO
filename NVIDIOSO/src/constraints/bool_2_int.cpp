//
//  bool_2_int.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "bool_2_int.h"

Bool2Int::Bool2Int ( std::string& constraint_name ) :
	BaseConstraint ( constraint_name ) {
	set_base_constraint_type ( BaseConstraintType::BOOL2INT );
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//Bool2Int

Bool2Int::~Bool2Int () {}

void
Bool2Int::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
Bool2Int::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
Bool2Int::consistency ()
{
}//consistency

//! It checks if
bool
Bool2Int::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
Bool2Int::print_semantic () const
{
    BaseConstraint::print_semantic ();
}//print_semantic



