//
//  base_constraint_register.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/20/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "base_constraint_register.h"
#include "constraint_inc.h"

using namespace std;

// -------------------------------------------------------------------- //
// -------------- POSTERs FUNCTIONs FOR BASE CONSTRAINTS -------------- //
// -------------------------------------------------------------------- //

BaseConstraint*  p_array_bool_and ( std::string& constraint_name ) 
{
	return new ArrayBoolAnd ( constraint_name );
}//p_array_bool_and

BaseConstraint*  p_array_bool_element ( std::string& constraint_name ) 
{
	return new ArrayBoolElement ( constraint_name );
}//p_array_bool_element

BaseConstraint*  p_array_bool_or ( std::string& constraint_name ) 
{
	return new ArrayBoolOr ( constraint_name );
}//p_array_bool_or

BaseConstraint*  p_array_int_element ( std::string& constraint_name ) 
{
	return new ArrayIntElement ( constraint_name );
}//p_array_int_element

BaseConstraint*  p_array_set_element ( std::string& constraint_name ) 
{
	return new ArraySetElement ( constraint_name );
}//p_array_set_element

BaseConstraint*  p_array_var_bool_element ( std::string& constraint_name ) 
{
	return new ArrayVarBoolElement ( constraint_name );
}//p_array_var_bool_element

BaseConstraint*  p_array_var_int_element ( std::string& constraint_name ) 
{
	return new ArrayVarIntElement ( constraint_name );
}//p_array_var_int_element

BaseConstraint*  p_array_var_set_element ( std::string& constraint_name ) 
{
	return new ArrayVarSetElement ( constraint_name );
}//p_array_var_set_element

BaseConstraint*  p_bool2int ( std::string& constraint_name ) 
{
	return new Bool2Int ( constraint_name );
}//p_bool2int

BaseConstraint*  p_bool_and ( std::string& constraint_name ) 
{
	return new BoolAnd ( constraint_name );
}//p_bool_and

BaseConstraint*  p_bool_clause ( std::string& constraint_name ) 
{
	return new BoolClause ( constraint_name );
}//p_bool_clause

BaseConstraint*  p_bool_eq ( std::string& constraint_name ) 
{
	return new BoolEq ( constraint_name );
}//p_bool_eq

BaseConstraint*  p_bool_eq_reif ( std::string& constraint_name ) 
{
	return new BoolEqReif ( constraint_name );
}//p_bool_eq_reif

BaseConstraint*  p_bool_le ( std::string& constraint_name ) 
{
	return new BoolLe ( constraint_name );
}//p_bool_le

BaseConstraint*  p_bool_le_reif ( std::string& constraint_name ) 
{
	return new BoolLeReif ( constraint_name );
}//p_bool_le_reif

BaseConstraint*  p_bool_lt ( std::string& constraint_name ) 
{
	return new BoolLt ( constraint_name );
}//p_bool_lt

BaseConstraint*  p_bool_lt_reif ( std::string& constraint_name ) 
{
	return new BoolLtReif ( constraint_name );
}//p_bool_lt_reif

BaseConstraint*  p_bool_not ( std::string& constraint_name ) 
{
	return new BoolNot ( constraint_name );
}//p_bool_not

BaseConstraint*  p_bool_or ( std::string& constraint_name ) 
{
	return new BoolOr ( constraint_name );
}//p_bool_or

BaseConstraint*  p_bool_xor ( std::string& constraint_name ) 
{
	return new BoolXor ( constraint_name );
}//p_bool_xor

BaseConstraint*  p_int_abs ( std::string& constraint_name ) 
{
	return new IntAbs ( constraint_name );
}//p_int_abs

BaseConstraint*  p_int_div ( std::string& constraint_name ) 
{
	return new IntDiv ( constraint_name );
}//p_int_div

BaseConstraint*  p_int_eq ( std::string& constraint_name ) 
{
	return new IntEq ( constraint_name );
}//p_int_eq

BaseConstraint*  p_int_eq_reif ( std::string& constraint_name ) 
{
	return new IntEqReif ( constraint_name );
}//p_int_eq_reif

BaseConstraint*  p_int_le ( std::string& constraint_name ) 
{
	return new IntLe ( constraint_name );
}//p_int_le

BaseConstraint*  p_int_le_reif ( std::string& constraint_name ) 
{
	return new IntLeReif ( constraint_name );
}//p_int_le_reif

BaseConstraint*  p_int_lin_eq ( std::string& constraint_name ) 
{
	return new IntLinEq ( constraint_name );
}//p_int_lin_eq

BaseConstraint*  p_int_lin_eq_reif ( std::string& constraint_name ) 
{
	return new IntLinEqReif ( constraint_name );
}//p_int_lin_eq_reif

BaseConstraint*  p_int_lin_le ( std::string& constraint_name ) 
{
	return new IntLinLe ( constraint_name );
}//p_int_lin_le

BaseConstraint*  p_int_lin_le_reif ( std::string& constraint_name ) 
{
	return new IntLinLeReif ( constraint_name );
}//p_int_lin_le_reif

BaseConstraint*  p_int_lin_ne ( std::string& constraint_name ) 
{
	return new IntLinNe ( constraint_name );
}//p_int_lin_ne

BaseConstraint*  p_int_lin_ne_reif ( std::string& constraint_name ) 
{
	return new IntLinNeReif ( constraint_name );
}//p_int_lin_ne_reif

BaseConstraint*  p_int_lt ( std::string& constraint_name ) 
{
	return new IntLt ( constraint_name );
}//p_int_lt

BaseConstraint*  p_int_lt_reif ( std::string& constraint_name ) 
{
	return new IntLtReif ( constraint_name );
}//p_int_lt_reif

BaseConstraint*  p_int_max ( std::string& constraint_name ) 
{
	return new IntMaxC ( constraint_name );
}//p_int_max

BaseConstraint*  p_int_min ( std::string& constraint_name ) 
{
	return new IntMinC ( constraint_name );
}//p_int_min

BaseConstraint*  p_int_mod ( std::string& constraint_name ) 
{
	return new IntMod ( constraint_name );
}//p_int_mod

BaseConstraint*  p_int_ne ( std::string& constraint_name ) 
{
	return new IntNe ( constraint_name );
}//p_int_ne

BaseConstraint*  p_int_ne_reif ( std::string& constraint_name ) 
{
	return new IntNeReif ( constraint_name );
}//p_int_ne_reif

BaseConstraint*  p_int_plus ( std::string& constraint_name ) 
{
	return new IntPlus ( constraint_name );
}//p_int_plus

BaseConstraint*  p_int_times ( std::string& constraint_name ) 
{
	return new IntTimes ( constraint_name );
}//p_int_times

BaseConstraint*  p_set_card ( std::string& constraint_name ) 
{
	return new SetCard ( constraint_name );
}//p_set_card

BaseConstraint*  p_set_diff ( std::string& constraint_name ) 
{
	return new SetDiff ( constraint_name );
}//p_set_diff

BaseConstraint*  p_set_eq ( std::string& constraint_name ) 
{
	return new SetEq ( constraint_name );
}//p_set_eq

BaseConstraint*  p_set_eq_reif ( std::string& constraint_name ) 
{
	return new SetEqReif ( constraint_name );
}//p_set_eq_reif

BaseConstraint*  p_set_in ( std::string& constraint_name ) 
{
	return new SetIn ( constraint_name );
}//p_set_in

BaseConstraint*  p_set_in_reif ( std::string& constraint_name ) 
{
	return new SetInReif ( constraint_name );
}//p_set_in_reif

BaseConstraint*  p_set_intersect ( std::string& constraint_name ) 
{
	return new SetIntersect ( constraint_name );
}//p_set_intersect

BaseConstraint*  p_set_le ( std::string& constraint_name ) 
{
	return new SetLe ( constraint_name );
}//p_set_le

BaseConstraint*  p_set_lt ( std::string& constraint_name ) 
{
	return new SetLt ( constraint_name );
}//p_set_lt

BaseConstraint*  p_set_ne ( std::string& constraint_name ) 
{
	return new SetNe ( constraint_name );
}//p_set_ne

BaseConstraint*  p_set_ne_reif ( std::string& constraint_name ) 
{
	return new SetNeReif ( constraint_name );
}//p_set_ne_reif

BaseConstraint*  p_set_subset ( std::string& constraint_name ) 
{
	return new SetSubset ( constraint_name );
}//p_set_subset

BaseConstraint*  p_set_subset_reif ( std::string& constraint_name ) 
{
	return new SetSubsetReif ( constraint_name );
}//p_set_subset_reif

BaseConstraint*  p_set_symdiff ( std::string& constraint_name ) 
{
	return new SetSymDiff ( constraint_name );
}//p_set_symdiff

BaseConstraint*  p_set_union ( std::string& constraint_name ) 
{
	return new SetUnion ( constraint_name );
}//p_set_union

// ---------------------------------------------------------------------- //
// ---------------------------------------------------------------------- //

BaseConstraintRegister::BaseConstraintRegister () {
	fill_register ();
}//GlobalConstraintRegister

BaseConstraintRegister::~BaseConstraintRegister () {
}//~GlobalConstraintRegister

void 
BaseConstraintRegister::add ( std::string name, poster p )
{
	_register [ name ] = p;
}//add

ConstraintPtr 
BaseConstraintRegister::get_base_constraint ( std::string& base_constraint_name )
{
	auto it = _register.find ( base_constraint_name );
	if ( it == _register.end () )
	{
		LogMsg << "BaseConstraintRegister::get_base_constraint - " << base_constraint_name << " not found." << endl;
		return nullptr;
	}
	
	// Create a new global constraint using the poster
	ConstraintPtr base_c = shared_ptr<BaseConstraint> ( _register [ base_constraint_name ] ( base_constraint_name ) );
	
	// Return the global constraint instance
	return base_c;
}//get_global_constraint

void 
BaseConstraintRegister::fill_register ()
{
	add ( "array_bool_and", p_array_bool_and );
	add ( "array_bool_element", p_array_bool_element );
	add ( "array_bool_or", p_array_bool_or );
	add ( "array_int_element", p_array_int_element );
	add ( "array_set_element", p_array_set_element );
	add ( "array_var_bool_element", p_array_var_bool_element );
	add ( "array_var_int_element", p_array_var_int_element );
	add ( "array_var_set_element", p_array_var_set_element );
	add ( "bool2int", p_bool2int );
	add ( "bool_and", p_bool_and );
	add ( "bool_clause", p_bool_clause );
	add ( "bool_eq", p_bool_eq );
	add ( "bool_eq_reif", p_bool_eq_reif );
	add ( "bool_le", p_bool_le );
	add ( "bool_le_reif", p_bool_le_reif );
	add ( "bool_lt", p_bool_lt );
	add ( "bool_lt_reif", p_bool_lt_reif );
	add ( "bool_not", p_bool_not );
	add ( "bool_or", p_bool_or );
	add ( "bool_xor", p_bool_xor );
	add ( "int_abs", p_int_abs);
	add ( "int_div", p_int_div );
	add ( "int_eq", p_int_eq );
	add ( "int_eq_reif", p_int_eq_reif );
	add ( "int_le", p_int_le );
	add ( "int_le_reif", p_int_le_reif );
	add ( "int_lin_eq", p_int_lin_eq );
	add ( "int_lin_eq_reif", p_int_lin_eq_reif );
	add ( "int_lin_le", p_int_lin_le );
	add ( "int_lin_le_reif", p_int_lin_le_reif );
	add ( "int_lin_ne", p_int_lin_ne );
	add ( "int_lin_ne_reif", p_int_lin_ne_reif );
	add ( "int_lt", p_int_lt );
	add ( "int_lt_reif", p_int_lt_reif );
	add ( "int_max", p_int_max );
	add ( "int_min", p_int_min );
	add ( "int_mod", p_int_mod);
	add ( "int_ne", p_int_ne );
	add ( "int_ne_reif", p_int_ne_reif );
	add ( "int_plus", p_int_plus );
	add ( "int_times", p_int_times );
	add ( "set_card", p_set_card );
	add ( "set_diff", p_set_diff );
	add ( "set_eq", p_set_eq );
	add ( "set_eq_reif", p_set_eq_reif );
	add ( "set_in", p_set_in );
	add ( "set_in_reif", p_set_in_reif );
	add ( "set_intersect", p_set_intersect );
	add ( "set_le", p_set_le );
	add ( "set_lt", p_set_lt );
	add ( "set_ne", p_set_ne );
	add ( "set_ne_reif", p_set_ne_reif );
	add ( "set_subset", p_set_subset );
	add ( "set_subset_reif", p_set_subset_reif );
	add ( "set_symdiff", p_set_symdiff );
	add ( "set_union", p_set_union );
}//fill_register