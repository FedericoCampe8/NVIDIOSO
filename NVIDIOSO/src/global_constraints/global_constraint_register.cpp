//
//  fzn_constraint.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/31/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "global_constraint_register.h"
#include "global_constraint_inc.h"

using namespace std;

// ---------------------------------------------------------------------- //
// -------------- POSTERs FUNCTIONs FOR GLOBAL CONSTRAINTS -------------- //
// ---------------------------------------------------------------------- //

GlobalConstraint*  p_alldifferent ( string constraint_name ) 
{
	return new Alldifferent ( constraint_name );
}//poster_alldifferent

// ---------------------------------------------------------------------- //
// ---------------------------------------------------------------------- //

GlobalConstraintRegister::GlobalConstraintRegister () {
	fill_register ();
}//GlobalConstraintRegister

GlobalConstraintRegister::~GlobalConstraintRegister () {
}//~GlobalConstraintRegister

void 
GlobalConstraintRegister::add ( std::string name, poster p )
{
	_register [ name ] = p;
}//add

GlobalConstraintPtr 
GlobalConstraintRegister::get_global_constraint ( std::string glb_constraint_name )
{
	auto it = _register.find ( glb_constraint_name );
	if ( it == _register.end () )
	{
		return nullptr;
	}
	
	// Create a new global constraint using the poster
	GlobalConstraintPtr glb_c = 
	make_shared<GlobalConstraint>( *(_register [ glb_constraint_name ] ( glb_constraint_name )) );
	
	// Return the global constraint instance
	return glb_c;
}//get_global_constraint

void 
GlobalConstraintRegister::fill_register ()
{
	add ( "alldifferent", p_alldifferent );
}//fill_register