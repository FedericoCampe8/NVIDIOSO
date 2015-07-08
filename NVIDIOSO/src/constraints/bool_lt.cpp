//
//  bool_lt.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "bool_lt.h"

BoolLt::BoolLt () :
FZNConstraint ( BOOL_LT ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntNe

BoolLt::BoolLt ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
BoolLt () {
  setup ( vars, args );
}//IntNe

BoolLt::~BoolLt () {}

void
BoolLt::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
BoolLt::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
BoolLt::consistency ()
{
}//consistency

//! It checks if
bool
BoolLt::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
BoolLt::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



