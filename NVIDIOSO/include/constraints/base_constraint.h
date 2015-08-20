//
//  base_constraint.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 28/07/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
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
//        - setup method
//        - consistency
//        - satisfied
//        - print_semantic
//  @note Remember to set the size of the scope in the constructor
//        according to the number of parameters given in input.
//  @note Remember to perform consistency checking on the pointers
//        given in input to each propagator.
//  @note See int_ne.h and int_ne.cpp for an example of how to write
//        a propagator.
//  @note If at least one FD variable involved in the constraint has
//        an empty domain and the rest of the variables satisfy the
//        the constraint, than the constraint is trivially satisfied.
//

#ifndef __NVIDIOSO__base_constraint__
#define __NVIDIOSO__base_constraint__

#include "constraint.h"

enum class BaseConstraintType
{
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
  INT_LT                  = 68,
  INT_LT_REIF             = 69,
  INT_MAX_C               = 70,
  INT_MIN_C               = 71,
  INT_MOD                 = 72,
  INT_NE                  = 73,
  INT_NE_REIF             = 74,
  INT_PLUS                = 75,
  INT_TIMES               = 76,
  INT2FLOAT               = 77,
  SET_CARD                = 78,
  SET_DIFF                = 79,
  SET_EQ                  = 80,
  SET_EQ_REIF             = 81,
  SET_IN                  = 82,
  SET_IN_REIF             = 83,
  SET_INTERSECT           = 84,
  SET_LE                  = 85,
  SET_LT                  = 86,
  SET_NE                  = 87,
  SET_NE_REIF             = 88,
  SET_SUBSET              = 89,
  SET_SUBSET_REIF         = 90,
  SET_SYMDIFF             = 91,
  SET_UNION               = 92,
  OTHER                   = 93
};

class BaseConstraint : public Constraint {
protected:
  
  //! FlatZinc constraint type
  BaseConstraintType _base_constraint_type;
  
  //! Scope size
  int _scope_size;
  
  //! Set base constraint type (default BaseConstraintType::OTHER )
  void set_base_constraint_type ( BaseConstraintType bse_t );
  
  /**
   * Base constructor.
   * @param name the name of the FlatZinc constraint.
   * @param vars the vector of (shared) pointers to the variables in the
   *        scope of this constraint.
   * @param args the vector of auxiliary arguments stored as strings
   *        needed by this constraint in order to be propagated.
   * @note FZNConstraint instantiated with this constructor
   *       need to be defined in terms of variables in their scope
   *       and, if needed, auxiliary parameters.
   */
  BaseConstraint ( std::string name );
  
public:
  
  virtual ~BaseConstraint();
  
  /**
   * It sets the variables and the arguments for this constraint.
   * @param vars a vector of pointers to the variables in the
   *        constraint's scope.
   * @param args a vector of strings representing the auxiliary
   *        arguments needed by the constraint in order to ensure
   *        consistency.
   */
  virtual void setup ( std::vector<VariablePtr> vars, std::vector<std::string> args );
  
  /**
   * It returns an integer value that can be used
   * to represent how much the current constraint is
   * unsatisfied. This function can be used to
   * implement some heuristics for optimization problems.
   * @return an integer value representing how much this 
   *         constraint is unsatisfied. It returns 0 if
   *         this constraint is satisfied.
   */
  virtual int unsat_level () const override;
  
  //! Get base constraint type
  BaseConstraintType get_base_constraint_type () const;
  
  /**
   * It attaches this constraint (observer) to the list of
   * the variables in its scope.
   * When a variable changes state, this constraint could be
   * automatically notified (depending on the variable).
   */
  void attach_me_to_vars () override;
  
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

#endif /* defined(__NVIDIOSO__base_constraint__) */
