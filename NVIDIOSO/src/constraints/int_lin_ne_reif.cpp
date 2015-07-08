//
//  int_lin_ne_reif.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "int_lin_ne_reif.h"

IntLinNeReif::IntLinNeReif () :
FZNConstraint ( INT_LIN_NE_REIF ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  //set_event( EventType::SINGLETON_EVT );
}//IntLinNeReif

IntLinNeReif::IntLinNeReif ( std::vector<VariablePtr> vars, std::vector<std::string> args ) :
IntLinNeReif () {
  setup ( vars, args );
}//IntLinNeReif

IntLinNeReif::~IntLinNeReif () {}

void
IntLinNeReif::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args )
{
}//setup

const std::vector<VariablePtr>
IntLinNeReif::scope () const
{
  // Return the constraint's scope
  std::vector<VariablePtr> scope;
  return scope;
}//scope

void
IntLinNeReif::consistency ()
{
}//consistency

//! It checks if
bool
IntLinNeReif::satisfied ()
{
  return true;
}//satisfied

//! Prints the semantic of this constraint
void
IntLinNeReif::print_semantic () const
{
    FZNConstraint::print_semantic ();
}//print_semantic



