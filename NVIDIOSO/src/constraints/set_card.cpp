//
//  set_card.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "set_card.h"

SetCard::SetCard () :
FZNConstraint ( SET_CARD ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//SetCard

SetCard::SetCard ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
SetCard () {
  setup ( vars, args );
}//SetCard

SetCard::~SetCard () {}

void
SetCard::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
SetCard::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
SetCard::consistency ()
{
}//consistency

//! It checks if
bool
SetCard::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
SetCard::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



