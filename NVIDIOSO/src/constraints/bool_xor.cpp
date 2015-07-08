//
//  bool_xor.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "bool_xor.h"

BoolXor::BoolXor () :
FZNConstraint ( BOOL_XOR ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntNe

BoolXor::BoolXor ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
BoolXor () {
  setup ( vars, args );
}//IntNe

BoolXor::~BoolXor () {}

void
BoolXor::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
BoolXor::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
BoolXor::consistency ()
{
}//consistency

//! It checks if
bool
BoolXor::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
BoolXor::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



