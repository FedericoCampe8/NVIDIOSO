//
//  fzn_constraint.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 28/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "fzn_constraint.h"

FZNConstraint::FZNConstraint ( std::string name ) :
Constraint () {
  _dbg             = name + " - ";
  _scope_size      = 0;
  _str_id          = name;
  _number_id       = name_to_id ( name );
  _constraint_type = int_to_type ( name_to_id ( name ) );
  
  // Set events for that will trigger the current constraint (default: CHANGE_EVT).
  set_events ();
}//FZNConstraint

void
FZNConstraint::set_events ( EventType event ) {
  _trigger_events.push_back ( event );
}//set_events

FZNConstraintType
FZNConstraint::int_to_type ( int number_id ) {
  
  // Consistency check
  if ( (number_id < 0) ||
       (number_id > type_to_int ( FZNConstraintType::OTHER )) ) {
    return FZNConstraintType::OTHER;
  }
  
  return static_cast<FZNConstraintType> ( number_id );
}//int_to_type

int
FZNConstraint::type_to_int ( FZNConstraintType c_type ) {
  return static_cast<int> ( c_type );
}//type_to_int

int
FZNConstraint::name_to_id ( std::string c_name ) {
  if ( c_name.compare ( ARRAY_BOOL_AND ) == 0 )
    return static_cast<int> (FZNConstraintType::ARRAY_BOOL_AND);
  if ( c_name.compare ( ARRAY_BOOL_ELEMENT ) == 0 )
    return static_cast<int> (FZNConstraintType::ARRAY_BOOL_ELEMENT);
  if ( c_name.compare ( ARRAY_FLOAT_ELEMENT ) == 0 )
    return static_cast<int> (FZNConstraintType::ARRAY_FLOAT_ELEMENT);
  if ( c_name.compare ( ARRAY_INT_ELEMENT ) == 0 )
    return static_cast<int> (FZNConstraintType::ARRAY_INT_ELEMENT);
  if ( c_name.compare ( ARRAY_SET_ELEMENT ) == 0 )
    return static_cast<int> (FZNConstraintType::ARRAY_SET_ELEMENT);
  if ( c_name.compare ( ARRAY_VAR_BOOL_ELEMENT ) == 0 )
    return static_cast<int> (FZNConstraintType::ARRAY_VAR_BOOL_ELEMENT);
  if ( c_name.compare ( ARRAY_VAR_FLOAT_ELEMENT ) == 0 )
    return static_cast<int> (FZNConstraintType::ARRAY_VAR_FLOAT_ELEMENT);
  if ( c_name.compare ( ARRAY_VAR_INT_ELEMENT ) == 0 )
    return static_cast<int> (FZNConstraintType::ARRAY_VAR_INT_ELEMENT);
  if ( c_name.compare ( ARRAY_VAR_SET_ELEMENT ) == 0 )
    return static_cast<int> (FZNConstraintType::ARRAY_VAR_SET_ELEMENT);
  if ( c_name.compare ( BOOL2INT ) == 0 )
    return static_cast<int> (FZNConstraintType::BOOL2INT);
  if ( c_name.compare ( BOOL_AND ) == 0 )
    return static_cast<int> (FZNConstraintType::BOOL_AND);
  if ( c_name.compare ( BOOL_CLAUSE ) == 0 )
    return static_cast<int> (FZNConstraintType::BOOL_CLAUSE);
  if ( c_name.compare ( BOOL_EQ ) == 0 )
    return static_cast<int> (FZNConstraintType::BOOL_EQ);
  if ( c_name.compare ( BOOL_EQ_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::BOOL_EQ_REIF);
  if ( c_name.compare ( BOOL_LE ) == 0 )
    return static_cast<int> (FZNConstraintType::BOOL_LE);
  if ( c_name.compare ( BOOL_LE_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::BOOL_LE_REIF);
  if ( c_name.compare ( BOOL_LT ) == 0 )
    return static_cast<int> (FZNConstraintType::BOOL_LT);
  if ( c_name.compare ( BOOL_LT_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::BOOL_LT_REIF);
  if ( c_name.compare ( BOOL_NOT ) == 0 )
    return static_cast<int> (FZNConstraintType::BOOL_NOT);
  if ( c_name.compare ( BOOL_OR ) == 0 )
    return static_cast<int> (FZNConstraintType::BOOL_OR);
  if ( c_name.compare ( BOOL_XOR ) == 0 )
    return static_cast<int> (FZNConstraintType::BOOL_XOR);
  if ( c_name.compare ( FLOAT_ABS ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_ABS);
  if ( c_name.compare ( FLOAT_ACOS ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_ACOS);
  if ( c_name.compare ( FLOAT_ASIN ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_ASIN);
  if ( c_name.compare ( FLOAT_ATAN ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_ATAN);
  if ( c_name.compare ( FLOAT_COS ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_COS);
  if ( c_name.compare ( FLOAT_COSH ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_COSH);
  if ( c_name.compare ( FLOAT_LN ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_LN);
  if ( c_name.compare ( FLOAT_LOG10 ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_LOG10);
  if ( c_name.compare ( FLOAT_LOG2 ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_LOG2);
  if ( c_name.compare ( FLOAT_SQRT ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_SQRT);
  if ( c_name.compare ( FLOAT_SIN ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_SIN);
  if ( c_name.compare ( FLOAT_SINH ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_SINH);
  if ( c_name.compare ( FLOAT_TAN ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_TAN);
  if ( c_name.compare ( FLOAT_TANH ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_TANH);
  if ( c_name.compare ( FLOAT_EQ ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_EQ);
  if ( c_name.compare ( FLOAT_EQ_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_EQ_REIF);
  if ( c_name.compare ( FLOAT_LE ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_LE);
  if ( c_name.compare ( FLOAT_LE_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_LE_REIF);
  if ( c_name.compare ( FLOAT_LIN_EQ ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_LIN_EQ);
  if ( c_name.compare ( FLOAT_LIN_EQ_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_LIN_EQ_REIF);
  if ( c_name.compare ( FLOAT_LIN_LE ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_LIN_LE);
  if ( c_name.compare ( FLOAT_LIN_LE_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_LIN_LE_REIF);
  if ( c_name.compare ( FLOAT_LIN_LT ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_LIN_LT);
  if ( c_name.compare ( FLOAT_LIN_LT_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_LIN_LT_REIF);
  if ( c_name.compare ( FLOAT_LIN_NE ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_LIN_NE);
  if ( c_name.compare ( FLOAT_LIN_NE_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_LIN_NE_REIF);
  if ( c_name.compare ( FLOAT_LT ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_LT);
  if ( c_name.compare ( FLOAT_LT_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_LT_REIF);
  if ( c_name.compare ( FLOAT_MAX ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_MAX);
  if ( c_name.compare ( FLOAT_MIN ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_MIN);
  if ( c_name.compare ( FLOAT_NE ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_NE);
  if ( c_name.compare ( FLOAT_NE_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_NE_REIF);
  if ( c_name.compare ( FLOAT_PLUS ) == 0 )
    return static_cast<int> (FZNConstraintType::FLOAT_PLUS);
  if ( c_name.compare ( INT_ABS ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_ABS);
  if ( c_name.compare ( INT_DIV ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_DIV);
  if ( c_name.compare ( INT_EQ ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_EQ);
  if ( c_name.compare ( INT_EQ_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_EQ_REIF);
  if ( c_name.compare ( INT_LE ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_LE);
  if ( c_name.compare ( INT_LE_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_LE_REIF);
  if ( c_name.compare ( INT_LIN_EQ ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_LIN_EQ);
  if ( c_name.compare ( INT_LIN_EQ_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_LIN_EQ_REIF);
  if ( c_name.compare ( INT_LIN_LE ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_LIN_LE);
  if ( c_name.compare ( INT_LIN_LE_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_LIN_LE_REIF);
  if ( c_name.compare ( INT_LIN_NE ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_LIN_NE);
  if ( c_name.compare ( INT_LIN_NE_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_LIN_NE_REIF);
  if ( c_name.compare ( INT_MAX_C ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_MAX_C);
  if ( c_name.compare ( INT_MIN_C ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_MIN_C);
  if ( c_name.compare ( INT_MOD ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_MOD);
  if ( c_name.compare ( INT_NE ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_NE);
  if ( c_name.compare ( INT_NE_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_NE_REIF);
  if ( c_name.compare ( INT_PLUS ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_PLUS);
  if ( c_name.compare ( INT_TIMES ) == 0 )
    return static_cast<int> (FZNConstraintType::INT_TIMES);
  if ( c_name.compare ( INT2FLOAT ) == 0 )
    return static_cast<int> (FZNConstraintType::INT2FLOAT);
  if ( c_name.compare ( SET_CARD ) == 0 )
    return static_cast<int> (FZNConstraintType::SET_CARD);
  if ( c_name.compare ( SET_DIFF ) == 0 )
    return static_cast<int> (FZNConstraintType::SET_DIFF);
  if ( c_name.compare ( SET_EQ ) == 0 )
    return static_cast<int> (FZNConstraintType::SET_EQ);
  if ( c_name.compare ( SET_EQ_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::SET_EQ_REIF);
  if ( c_name.compare ( SET_IN ) == 0 )
    return static_cast<int> (FZNConstraintType::SET_IN);
  if ( c_name.compare ( SET_IN_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::SET_IN_REIF);
  if ( c_name.compare ( SET_INTERSECT ) == 0 )
    return static_cast<int> (FZNConstraintType::SET_INTERSECT);
  if ( c_name.compare ( SET_LE ) == 0 )
    return static_cast<int> (FZNConstraintType::SET_LE);
  if ( c_name.compare ( SET_LT ) == 0 )
    return static_cast<int> (FZNConstraintType::SET_LT);
  if ( c_name.compare ( SET_NE ) == 0 )
    return static_cast<int> (FZNConstraintType::SET_NE);
  if ( c_name.compare ( SET_NE_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::SET_NE_REIF);
  if ( c_name.compare ( SET_SUBSET ) == 0 )
    return static_cast<int> (FZNConstraintType::SET_SUBSET);
  if ( c_name.compare ( SET_SUBSET_REIF ) == 0 )
    return static_cast<int> (FZNConstraintType::SET_SUBSET_REIF);
  if ( c_name.compare ( SET_SYMDIFF ) == 0 )
    return static_cast<int> (FZNConstraintType::SET_SYMDIFF);
  if ( c_name.compare ( SET_UNION ) == 0 )
    return static_cast<int> (FZNConstraintType::SET_UNION);
  
  // Return other
  return static_cast<int> (FZNConstraintType::OTHER);
}//name_to_id

void
FZNConstraint::attach_me () {
  for ( auto var : scope() )
    var->attach_constraint( get_this_shared_ptr() );
}//attach_me

void
FZNConstraint::consistency () {
  throw NvdException ( (_dbg + "Constraint " + _str_id + " not yet implemented").c_str() );
}//consistency

bool
FZNConstraint::satisfied () {
  throw NvdException ( (_dbg + "Constraint " + _str_id + " not yet implemented").c_str() );
}//satisfied

void
FZNConstraint::remove_constraint () {
  for ( auto var : scope() ) {
    var->detach_constraint ( get_unique_id() );
  }
}//remove_constraint

void
FZNConstraint::print () const {
  std::cout << "Constraint_" << get_unique_id () << ": " << _str_id <<
  "\t (FlatZinc id: " << _number_id << ")\n";
  std::cout << "Scope size: " << get_scope_size() << "\n";
  std::cout << "Weight:     " << get_weight ()     << "\n";
  if ( get_scope_size() ) {
  std::cout << "Variables:\n";
    for ( auto var : scope() ) std::cout << var->get_str_id() << " ";
    std::cout << std::endl;
  }
  if ( _arguments.size() ) {
    std::cout << "Arguments:\n";
    for ( auto arg : _arguments ) std::cout << arg << " ";
    std::cout << std::endl;
  }
}//print

void
FZNConstraint::print_semantic () const {
  std::cout << "Semantic:\n";
}//print_semantic

const std::string FZNConstraint::ARRAY_BOOL_AND          = "array_bool_and";
const std::string FZNConstraint::ARRAY_BOOL_ELEMENT      = "array_bool_element";
const std::string FZNConstraint::ARRAY_BOOL_OR           = "array_bool_or";
const std::string FZNConstraint::ARRAY_FLOAT_ELEMENT     = "array_float_element";
const std::string FZNConstraint::ARRAY_INT_ELEMENT       = "array_int_element";
const std::string FZNConstraint::ARRAY_SET_ELEMENT       = "array_set_element";
const std::string FZNConstraint::ARRAY_VAR_BOOL_ELEMENT  = "array_var_bool_element";
const std::string FZNConstraint::ARRAY_VAR_FLOAT_ELEMENT = "array_var_float_element";
const std::string FZNConstraint::ARRAY_VAR_INT_ELEMENT   = "array_var_int_element";
const std::string FZNConstraint::ARRAY_VAR_SET_ELEMENT   = "array_var_set_element";
const std::string FZNConstraint::BOOL2INT                = "bool2int";
const std::string FZNConstraint::BOOL_AND                = "bool_and";
const std::string FZNConstraint::BOOL_CLAUSE             = "bool_clause";
const std::string FZNConstraint::BOOL_EQ                 = "bool_eq";
const std::string FZNConstraint::BOOL_EQ_REIF            = "bool_eq_reif";
const std::string FZNConstraint::BOOL_LE                 = "bool_le";
const std::string FZNConstraint::BOOL_LE_REIF            = "bool_le_reif";
const std::string FZNConstraint::BOOL_LT                 = "bool_lt";
const std::string FZNConstraint::BOOL_LT_REIF            = "bool_lt_reif";
const std::string FZNConstraint::BOOL_NOT                = "bool_not";
const std::string FZNConstraint::BOOL_OR                 = "bool_or";
const std::string FZNConstraint::BOOL_XOR                = "bool_xor";
const std::string FZNConstraint::FLOAT_ABS               = "float_abs";
const std::string FZNConstraint::FLOAT_ACOS              = "float_acos";
const std::string FZNConstraint::FLOAT_ASIN              = "float_asin";
const std::string FZNConstraint::FLOAT_ATAN              = "float_atan";
const std::string FZNConstraint::FLOAT_COS               = "float_cos";
const std::string FZNConstraint::FLOAT_COSH              = "float_cosh";
const std::string FZNConstraint::FLOAT_EXP               = "float_exp";
const std::string FZNConstraint::FLOAT_LN                = "float_ln";
const std::string FZNConstraint::FLOAT_LOG10             = "float_log10";
const std::string FZNConstraint::FLOAT_LOG2              = "float_log2";
const std::string FZNConstraint::FLOAT_SQRT              = "float_sqrt";
const std::string FZNConstraint::FLOAT_SIN               = "float_sin";
const std::string FZNConstraint::FLOAT_SINH              = "float_sinh";
const std::string FZNConstraint::FLOAT_TAN               = "float_tan";
const std::string FZNConstraint::FLOAT_TANH              = "float_tanh";
const std::string FZNConstraint::FLOAT_EQ                = "float_eq";
const std::string FZNConstraint::FLOAT_EQ_REIF           = "float_eq_reif";
const std::string FZNConstraint::FLOAT_LE                = "float_le";
const std::string FZNConstraint::FLOAT_LE_REIF           = "float_le_reif";
const std::string FZNConstraint::FLOAT_LIN_EQ            = "float_lin_eq";
const std::string FZNConstraint::FLOAT_LIN_EQ_REIF       = "float_lin_eq_reif";
const std::string FZNConstraint::FLOAT_LIN_LE            = "float_lin_le";
const std::string FZNConstraint::FLOAT_LIN_LE_REIF       = "float_lin_le_reif";
const std::string FZNConstraint::FLOAT_LIN_LT            = "float_lin_lt";
const std::string FZNConstraint::FLOAT_LIN_LT_REIF       = "float_lin_lt_reif";
const std::string FZNConstraint::FLOAT_LIN_NE            = "float_lin_ne";
const std::string FZNConstraint::FLOAT_LIN_NE_REIF       = "float_lin_ne_reif";
const std::string FZNConstraint::FLOAT_LT                = "float_lt";
const std::string FZNConstraint::FLOAT_LT_REIF           = "float_lt_reif";
const std::string FZNConstraint::FLOAT_MAX               = "float_max";
const std::string FZNConstraint::FLOAT_MIN               = "float_min";
const std::string FZNConstraint::FLOAT_NE                = "float_ne";
const std::string FZNConstraint::FLOAT_NE_REIF           = "float_ne_reif";
const std::string FZNConstraint::FLOAT_PLUS              = "float_plus";
const std::string FZNConstraint::INT_ABS                 = "int_abs";
const std::string FZNConstraint::INT_DIV                 = "int_div";
const std::string FZNConstraint::INT_EQ                  = "int_eq";
const std::string FZNConstraint::INT_EQ_REIF             = "int_eq_reif";
const std::string FZNConstraint::INT_LE                  = "int_le";
const std::string FZNConstraint::INT_LE_REIF             = "int_le_reif";
const std::string FZNConstraint::INT_LIN_EQ              = "int_lin_eq";
const std::string FZNConstraint::INT_LIN_EQ_REIF         = "int_lin_eq_reif";
const std::string FZNConstraint::INT_LIN_LE              = "int_lin_le";
const std::string FZNConstraint::INT_LIN_LE_REIF         = "int_lin_le_reif";
const std::string FZNConstraint::INT_LIN_NE              = "int_lin_ne";
const std::string FZNConstraint::INT_LIN_NE_REIF         = "int_lin_ne_reif";
const std::string FZNConstraint::INT_MAX_C               = "int_max";
const std::string FZNConstraint::INT_MIN_C               = "int_min";
const std::string FZNConstraint::INT_MOD                 = "int_mod";
const std::string FZNConstraint::INT_NE                  = "int_ne";
const std::string FZNConstraint::INT_NE_REIF             = "int_ne_reif";
const std::string FZNConstraint::INT_PLUS                = "int_plus";
const std::string FZNConstraint::INT_TIMES               = "int_times";
const std::string FZNConstraint::INT2FLOAT               = "int2float";
const std::string FZNConstraint::SET_CARD                = "set_card";
const std::string FZNConstraint::SET_DIFF                = "set_diff";
const std::string FZNConstraint::SET_EQ                  = "set_eq";
const std::string FZNConstraint::SET_EQ_REIF             = "set_eq_reif";
const std::string FZNConstraint::SET_IN                  = "set_in";
const std::string FZNConstraint::SET_IN_REIF             = "set_in_reif";
const std::string FZNConstraint::SET_INTERSECT           = "set_intersect";
const std::string FZNConstraint::SET_LE                  = "set_le";
const std::string FZNConstraint::SET_LT                  = "set_lt";
const std::string FZNConstraint::SET_NE                  = "set_ne";
const std::string FZNConstraint::SET_NE_REIF             = "set_ne_reif";
const std::string FZNConstraint::SET_SUBSET              = "set_subset";
const std::string FZNConstraint::SET_SUBSET_REIF         = "set_subset_reif";
const std::string FZNConstraint::SET_SYMDIFF             = "set_symdiff";
const std::string FZNConstraint::SET_UNION               = "set_union";
const std::string FZNConstraint::OTHER                   = "other";
