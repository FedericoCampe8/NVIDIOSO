//
//  int_lin_le.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "int_lin_le.h"

IntLinLe::IntLinLe () :
FZNConstraint ( INT_LIN_LE ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntLinLe

IntLinLe::IntLinLe ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
IntLinLe () {
  setup ( vars, args );
}//IntLinLe

IntLinLe::~IntLinLe () {}

void
IntLinLe::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
IntLinLe::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
IntLinLe::consistency ()
{
}//consistency

//! It checks if
bool
IntLinLe::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
IntLinLe::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



