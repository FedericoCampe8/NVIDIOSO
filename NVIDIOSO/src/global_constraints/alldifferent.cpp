//
//  int_ne.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 29/07/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "alldifferent.h"

Alldifferent::Alldifferent ( std::string constraint_name ) :
GlobalConstraint ( constraint_name ) {
  /*
   * Set the event that trigger this constraint.
   * @note if no event is set, this constraint will never be re-evaluated.
   */
  set_event( EventType::CHANGE_EVT );
  
  // Set constraint type
  set_global_constraint_type ( GlobalConstraintType::ALLDIFFERENT );
}//Alldifferent

Alldifferent::~Alldifferent () {
}//~Alldifferent

void
Alldifferent::setup ( std::vector<VariablePtr> vars, std::vector<std::string> args ) 
{  
}//setup

void
Alldifferent::consistency () 
{
	return;
}//consistency

bool
Alldifferent::satisfied ()  
{
  return true;
}//satisfied

void 
Alldifferent::print () const 
{
	print_semantic ();
}//print

//! Prints the semantic of this constraint
void
Alldifferent::print_semantic () const 
{
	GlobalConstraint::print_semantic();
}//print_semantic



