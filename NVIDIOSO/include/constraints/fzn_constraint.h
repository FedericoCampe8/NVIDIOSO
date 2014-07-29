//
//  fzn_constraint.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 28/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class specializes the Constraint class with the constraints
//  as defined by the Specification of the FlatZinc language.
//  The interested reader can consult
//      http://www.minizinc.org/downloads/doc-1.3/flatzinc-spec.pdf
//  for the complete list of constraints and their specification.
//
//  @note To implement a constraint at least the following methods
//        should be specialized:
//        - constructor
//        - consistency
//        - satisfied
//        - print_semantic
//  @note Remember to set the size of the scope in the constructor
//        according to the number of parameters given in input.
//  @note Remember to perform consistency checking on the pointers
//        given in input to each propagator.
//  @note See int_ne.h and int_ne.cpp for an example of how to write
//        a propagator.
//

#ifndef __NVIDIOSO__fzn_constraint__
#define __NVIDIOSO__fzn_constraint__

#include "constraint.h"

enum class FZNConstraintType {
  ARRAY_BOOL_AND          = 0,
  ARRAY_BOOL_ELEMENT      = 1,
  ARRAY_BOOL_OR           = 2,
  ARRAY_FLOAT_ELEMENT     = 3,
  ARRAY_INT_ELEMENT       = 4,
  ARRAY_SET_ELEMENT       = 5,
  ARRAY_VAR_BOOL_ELEMENT  = 6,
  ARRAY_VAR_FLOAT_ELEMENT = 7,
  ARRAY_VAR_INT_ELEMENT   = 8,
  ARRAY_VAR_SET_ELEMENT   = 9,
  BOOL2INT                = 10,
  BOOL_AND                = 11,
  BOOL_CLAUSE             = 12,
  BOOL_EQ                 = 13,
  BOOL_EQ_REIF            = 14,
  BOOL_LE                 = 15,
  BOOL_LE_REIF            = 16,
  BOOL_LT                 = 17,
  BOOL_LT_REIF            = 18,
  BOOL_NOT                = 19,
  BOOL_OR                 = 20,
  BOOL_XOR                = 21,
  FLOAT_ABS               = 22,
  FLOAT_ACOS              = 23,
  FLOAT_ASIN              = 24,
  FLOAT_ATAN              = 25,
  FLOAT_COS               = 26,
  FLOAT_COSH              = 27,
  FLOAT_EXP               = 28,
  FLOAT_LN                = 29,
  FLOAT_LOG10             = 30,
  FLOAT_LOG2              = 31,
  FLOAT_SQRT              = 32,
  FLOAT_SIN               = 33,
  FLOAT_SINH              = 34,
  FLOAT_TAN               = 35,
  FLOAT_TANH              = 36,
  FLOAT_EQ                = 37,
  FLOAT_EQ_REIF           = 38,
  FLOAT_LE                = 39,
  FLOAT_LE_REIF           = 40,
  FLOAT_LIN_EQ            = 41,
  FLOAT_LIN_EQ_REIF       = 42,
  FLOAT_LIN_LE            = 43,
  FLOAT_LIN_LE_REIF       = 44,
  FLOAT_LIN_LT            = 45,
  FLOAT_LIN_LT_REIF       = 46,
  FLOAT_LIN_NE            = 47,
  FLOAT_LIN_NE_REIF       = 48,
  FLOAT_LT                = 49,
  FLOAT_LT_REIF           = 50,
  FLOAT_MAX               = 51,
  FLOAT_MIN               = 52,
  FLOAT_NE                = 53,
  FLOAT_NE_REIF           = 54,
  FLOAT_PLUS              = 55,
  INT_ABS                 = 56,
  INT_DIV                 = 57,
  INT_EQ                  = 58,
  INT_EQ_REIF             = 59,
  INT_LE                  = 60,
  INT_LE_REIF             = 61,
  INT_LIN_EQ              = 62,
  INT_LIN_EQ_REIF         = 63,
  INT_LIN_LE              = 64,
  INT_LIN_LE_REIF         = 65,
  INT_LIN_NE              = 66,
  INT_LIN_NE_REIF         = 67,
  INT_MAX_C               = 68,
  INT_MIN_C               = 69,
  INT_MOD                 = 70,
  INT_NE                  = 71,
  INT_NE_REIF             = 72,
  INT_PLUS                = 73,
  INT_TIMES               = 74,
  INT2FLOAT               = 75,
  SET_CARD                = 76,
  SET_DIFF                = 77,
  SET_EQ                  = 78,
  SET_EQ_REIF             = 79,
  SET_IN                  = 80,
  SET_IN_REIF             = 81,
  SET_INTERSECT           = 82,
  SET_LE                  = 83,
  SET_LT                  = 84,
  SET_NE                  = 85,
  SET_NE_REIF             = 86,
  SET_SUBSET              = 87,
  SET_SUBSET_REIF         = 88,
  SET_SYMDIFF             = 89,
  SET_UNION               = 90,
  OTHER                   = 91
};

class FZNConstraint : public Constraint {
protected:
  const std::string ARRAY_BOOL_AND          = "array_bool_and";
  const std::string ARRAY_BOOL_ELEMENT      = "array_bool_element";
  const std::string ARRAY_BOOL_OR           = "array_bool_or";
  const std::string ARRAY_FLOAT_ELEMENT     = "array_float_element";
  const std::string ARRAY_INT_ELEMENT       = "array_int_element";
  const std::string ARRAY_SET_ELEMENT       = "array_set_element";
  const std::string ARRAY_VAR_BOOL_ELEMENT  = "array_var_bool_element";
  const std::string ARRAY_VAR_FLOAT_ELEMENT = "array_var_float_element";
  const std::string ARRAY_VAR_INT_ELEMENT   = "array_var_int_element";
  const std::string ARRAY_VAR_SET_ELEMENT   = "array_var_set_element";
  const std::string BOOL2INT                = "bool2int";
  const std::string BOOL_AND                = "bool_and";
  const std::string BOOL_CLAUSE             = "bool_clause";
  const std::string BOOL_EQ                 = "bool_eq";
  const std::string BOOL_EQ_REIF            = "bool_eq_reif";
  const std::string BOOL_LE                 = "bool_le";
  const std::string BOOL_LE_REIF            = "bool_le_reif";
  const std::string BOOL_LT                 = "bool_lt";
  const std::string BOOL_LT_REIF            = "bool_lt_reif";
  const std::string BOOL_NOT                = "bool_not";
  const std::string BOOL_OR                 = "bool_or";
  const std::string BOOL_XOR                = "bool_xor";
  const std::string FLOAT_ABS               = "float_abs";
  const std::string FLOAT_ACOS              = "float_acos";
  const std::string FLOAT_ASIN              = "float_asin";
  const std::string FLOAT_ATAN              = "float_atan";
  const std::string FLOAT_COS               = "float_cos";
  const std::string FLOAT_COSH              = "float_cosh";
  const std::string FLOAT_EXP               = "float_exp";
  const std::string FLOAT_LN                = "float_ln";
  const std::string FLOAT_LOG10             = "float_log10";
  const std::string FLOAT_LOG2              = "float_log2";
  const std::string FLOAT_SQRT              = "float_sqrt";
  const std::string FLOAT_SIN               = "float_sin";
  const std::string FLOAT_SINH              = "float_sinh";
  const std::string FLOAT_TAN               = "float_tan";
  const std::string FLOAT_TANH              = "float_tanh";
  const std::string FLOAT_EQ                = "float_eq";
  const std::string FLOAT_EQ_REIF           = "float_eq_reif";
  const std::string FLOAT_LE                = "float_le";
  const std::string FLOAT_LE_REIF           = "float_le_reif";
  const std::string FLOAT_LIN_EQ            = "float_lin_eq";
  const std::string FLOAT_LIN_EQ_REIF       = "float_lin_eq_reif";
  const std::string FLOAT_LIN_LE            = "float_lin_le";
  const std::string FLOAT_LIN_LE_REIF       = "float_lin_le_reif";
  const std::string FLOAT_LIN_LT            = "float_lin_lt";
  const std::string FLOAT_LIN_LT_REIF       = "float_lin_lt_reif";
  const std::string FLOAT_LIN_NE            = "float_lin_ne";
  const std::string FLOAT_LIN_NE_REIF       = "float_lin_ne_reif";
  const std::string FLOAT_LT                = "float_lt";
  const std::string FLOAT_LT_REIF           = "float_lt_reif";
  const std::string FLOAT_MAX               = "float_max";
  const std::string FLOAT_MIN               = "float_min";
  const std::string FLOAT_NE                = "float_ne";
  const std::string FLOAT_NE_REIF           = "float_ne_reif";
  const std::string FLOAT_PLUS              = "float_plus";
  const std::string INT_ABS                 = "int_abs";
  const std::string INT_DIV                 = "int_div";
  const std::string INT_EQ                  = "int_eq";
  const std::string INT_EQ_REIF             = "int_eq_reif";
  const std::string INT_LE                  = "int_le";
  const std::string INT_LE_REIF             = "int_le_reif";
  const std::string INT_LIN_EQ              = "int_lin_eq";
  const std::string INT_LIN_EQ_REIF         = "int_lin_eq_reif";
  const std::string INT_LIN_LE              = "int_lin_le";
  const std::string INT_LIN_LE_REIF         = "int_lin_le_reif";
  const std::string INT_LIN_NE              = "int_lin_ne";
  const std::string INT_LIN_NE_REIF         = "int_lin_ne_reif";
  const std::string INT_MAX_C               = "int_max";
  const std::string INT_MIN_C               = "int_min";
  const std::string INT_MOD                 = "int_mod";
  const std::string INT_NE                  = "int_ne";
  const std::string INT_NE_REIF             = "int_ne_reif";
  const std::string INT_PLUS                = "int_plus";
  const std::string INT_TIMES               = "int_times";
  const std::string INT2FLOAT               = "int2float";
  const std::string SET_CARD                = "set_card";
  const std::string SET_DIFF                = "set_diff";
  const std::string SET_EQ                  = "set_eq";
  const std::string SET_EQ_REIF             = "set_eq_reif";
  const std::string SET_IN                  = "set_in";
  const std::string SET_IN_REIF             = "set_in_reif";
  const std::string SET_INTERSECT           = "set_intersect";
  const std::string SET_LE                  = "set_le";
  const std::string SET_LT                  = "set_lt";
  const std::string SET_NE                  = "set_ne";
  const std::string SET_NE_REIF             = "set_ne_reif";
  const std::string SET_SUBSET              = "set_subset";
  const std::string SET_SUBSET_REIF         = "set_subset_reif";
  const std::string SET_SYMDIFF             = "set_symdiff";
  const std::string SET_UNION               = "set_union";
  const std::string OTHER                   = "other";
  
  //! FlatZinc constraint type
  FZNConstraintType _constraint_type;
  
  //! Scope size
  int _scope_size;
  
  /**
   * It converts a number_id name to the
   * correspondent FZNConstraintType type.
   * @param  number_id the number id of the FlatZinc constraint.
   * @return the type of the FlatZinc constraint.
   */
  FZNConstraintType int_to_type ( int number_id ) const;
  
  /**
   * It converts a FZNConstraintType to the
   * correspondent integer type.
   * @param  c_type the type of the FlatZinc constraint.
   * @return the number_id correspondent to c_type.
   */
  int type_to_int ( FZNConstraintType c_type ) const;
  
  /**
   * It converts a string representing the name of a constraint
   * to a unique idetifier for the correspondent type
   * of FlatZinc constraint.
   * @param c_name name of a FlatZinc constraint.
   * @return the number_id correspondent to name.
   */
  int name_to_id ( std::string c_name ) const;
  
  /**
   * Set the events that trigger this constraint.
   * @note default: CHANGE_EVT.
   * @note different constraints should specilize this method
   *       with the appropriate list of events.
   */
  virtual void set_events ( EventType event = EventType::CHANGE_EVT );
  
public:
  /**
   * Base constructor.
   * @param name the name of the FlatZinc constraint.
   * @note FZNConstraint instantiated with this constructor
   *       need to be defined in terms of variables in their scope
   *       and, if needed, auxiliary parameters.
   */
  FZNConstraint ( std::string name );
  
  /**
   * Constructor for FZNConstraint constraints.
   * @param name the name of the FlatZinc constraint.
   * @param scope_vars the array of (pointers to) variables within
   *        the scope of this constraint.
   */
  FZNConstraint ( std::string name, std::vector<VariablePtr> scope_vars );
  
  /**
   * Constructor for FZNConstraint constraints.
   * @param name the name of the FlatZinc constraint.
   * @param scope_vars the array of (pointers to) variables within
   *        the scope of this constraint.
   * @param auxiliary_params the array of integers representing
   *        the auxiliary parameters needed for this constraint
   *        in order to be propagated on the variables in its scope.
   */
  FZNConstraint ( std::string name,
                  std::vector<VariablePtr> scope_vars,
                  std::vector<int> auxiliary_params );
  
  /**
   * It is a (most probably incomplete) consistency function which
   * removes the values from variable domains. Only values which
   * do not have any support in a solution space are removed.
   */
  void consistency () override;
  
  /**
   * It checks if the constraint is satisfied.
   * @return true if the constraint if for certain satisfied,
   *         false otherwise.
   * @note If this function is incorrectly implementd,
   *       a constraint may not be satisfied in a solution.
   */
  bool satisfied () override;
  
  /**
   * It removes the constraint by removing this constraint
   * from all variables in its scope.
   */
  void remove_constraint ();
  
  //! Prints info.
  void print () const override;
  
  //! Prints the semantic of this constraint.
  void print_semantic () const override;
};

#endif /* defined(__NVIDIOSO__fzn_constraint__) */
