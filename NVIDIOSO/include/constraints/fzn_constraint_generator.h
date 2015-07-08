//
//  fzn_constraint_generator.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 30/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class is a factory class for generating instances of
//  FlatZinc constraints according to the type described by the string
//  representing the constraint as parsed from the FlatZinc input model.
//

#ifndef NVIDIOSO_fzn_constraint_generator_h
#define NVIDIOSO_fzn_constraint_generator_h

#include "constraint_inc.h"

class FZNConstraintFactory {
public:
  
  /**
   * Get the right instance of FlatZinc constraint
   * according to its type described by the input string.
   * @param c_name the FlatZinc name of the constraint to instantiate.
   * @param vars the vector of (shared) pointer to the FD variables in the
   *        scope of the constraint to instantiate.
   * @param args the vector of strings representing the auxiliary arguments
   *        needed by the constraint to instantiate in order to be propagated.
   */
  static ConstraintPtr get_fzn_constraint_shr_ptr ( std::string c_name,
                                                    std::vector<VariablePtr> vars,
                                                    std::vector<std::string> args ) {
    
    
    
    FZNConstraintType c_type =
    FZNConstraint::int_to_type( FZNConstraint::name_to_id ( c_name ) );
    
    switch ( c_type )
    {
        case FZNConstraintType::ARRAY_BOOL_AND:
            return std::make_shared<ArrayBoolAnd>( vars, args );
        case FZNConstraintType::ARRAY_BOOL_ELEMENT:
            return std::make_shared<ArrayBoolElement>( vars, args );
        case FZNConstraintType::ARRAY_BOOL_OR:
            return std::make_shared<ArrayBoolOr>( vars, args );
        case FZNConstraintType::ARRAY_INT_ELEMENT:
            return std::make_shared<ArrayIntElement>( vars, args );
        case FZNConstraintType::ARRAY_SET_ELEMENT:
            return std::make_shared<ArraySetElement>( vars, args );
        case FZNConstraintType::ARRAY_VAR_BOOL_ELEMENT:
            return std::make_shared<ArrayVarBoolElement>( vars, args );
        case FZNConstraintType::ARRAY_VAR_INT_ELEMENT:
            return std::make_shared<ArrayVarIntElement>( vars, args );
        case FZNConstraintType::ARRAY_VAR_SET_ELEMENT:
            return std::make_shared<ArrayVarSetElement>( vars, args );
        case FZNConstraintType::BOOL2INT:
            return std::make_shared<Bool2Int>( vars, args );
        case FZNConstraintType::BOOL_AND:
            return std::make_shared<BoolAnd>( vars, args );
        case FZNConstraintType::BOOL_CLAUSE:
            return std::make_shared<BoolClause>( vars, args );
        case FZNConstraintType::BOOL_EQ:
            return std::make_shared<BoolEq>( vars, args );
        case FZNConstraintType::BOOL_EQ_REIF:
            return std::make_shared<BoolEqReif>( vars, args );
        case FZNConstraintType::BOOL_LE:
            return std::make_shared<BoolLe>( vars, args );
        case FZNConstraintType::BOOL_LE_REIF:
            return std::make_shared<BoolLeReif>( vars, args );
        case FZNConstraintType::BOOL_LT:
            return std::make_shared<BoolLt>( vars, args );
        case FZNConstraintType::BOOL_LT_REIF:
            return std::make_shared<BoolLtReif>( vars, args );
        case FZNConstraintType::BOOL_NOT:
            return std::make_shared<BoolNot>( vars, args );
        case FZNConstraintType::BOOL_OR:
            return std::make_shared<BoolOr>( vars, args );
        case FZNConstraintType::BOOL_XOR:
            return std::make_shared<BoolXor>( vars, args );
        case FZNConstraintType::INT_ABS:
            return std::make_shared<IntAbs>( vars, args );
        case FZNConstraintType::INT_DIV:
            return std::make_shared<IntDiv>( vars, args );
        case FZNConstraintType::INT_EQ:
            return std::make_shared<IntEq>( vars, args );
        case FZNConstraintType::INT_EQ_REIF:
            return std::make_shared<IntEqReif>( vars, args );
        case FZNConstraintType::INT_LE:
            return std::make_shared<IntLe>( vars, args );
        case FZNConstraintType::INT_LE_REIF:
            return std::make_shared<IntLeReif>( vars, args );
        case FZNConstraintType::INT_LIN_EQ:
            return std::make_shared<IntLinEq>( vars, args );
        case FZNConstraintType::INT_LIN_EQ_REIF:
            return std::make_shared<IntLinEqReif>( vars, args );
        case FZNConstraintType::INT_LIN_LE:
            return std::make_shared<IntLinLe>( vars, args );
        case FZNConstraintType::INT_LIN_LE_REIF:
            return std::make_shared<IntLinLeReif>( vars, args );
        case FZNConstraintType::INT_LIN_NE:
            return std::make_shared<IntLinNe>( vars, args );
        case FZNConstraintType::INT_LIN_NE_REIF:
            return std::make_shared<IntLinNeReif>( vars, args );
        case FZNConstraintType::INT_LT:
            return std::make_shared<IntLt>( vars, args );
        case FZNConstraintType::INT_LT_REIF:
            return std::make_shared<IntLtReif>( vars, args );
        case FZNConstraintType::INT_MAX_C:
            return std::make_shared<IntMaxC>( vars, args );
        case FZNConstraintType::INT_MIN_C:
            return std::make_shared<IntMinC>( vars, args );
        case FZNConstraintType::INT_MOD:
            return std::make_shared<IntMod>( vars, args );
        case FZNConstraintType::INT_NE:
            return std::make_shared<IntNe>( vars, args );
        case FZNConstraintType::INT_NE_REIF:
            return std::make_shared<IntNeReif>( vars, args );
        case FZNConstraintType::INT_PLUS:
            return std::make_shared<IntPlus>( vars, args );
        case FZNConstraintType::INT_TIMES:
            return std::make_shared<IntTimes>( vars, args );
        case FZNConstraintType::SET_CARD:
            return std::make_shared<SetCard>( vars, args );
        case FZNConstraintType::SET_DIFF:
            return std::make_shared<SetDiff>( vars, args );
        case FZNConstraintType::SET_EQ:
            return std::make_shared<SetEq>( vars, args );
        case FZNConstraintType::SET_EQ_REIF:
            return std::make_shared<SetEqReif>( vars, args );
        case FZNConstraintType::SET_IN:
            return std::make_shared<SetIn>( vars, args );
        case FZNConstraintType::SET_IN_REIF:
            return std::make_shared<SetInReif>( vars, args );
        case FZNConstraintType::SET_INTERSECT:
            return std::make_shared<SetIntersect>( vars, args );
        case FZNConstraintType::SET_LE:
            return std::make_shared<SetLe>( vars, args );
        case FZNConstraintType::SET_LT:
            return std::make_shared<SetLt>( vars, args );
        case FZNConstraintType::SET_NE:
            return std::make_shared<SetNe>( vars, args );
        case FZNConstraintType::SET_NE_REIF:
            return std::make_shared<SetNeReif>( vars, args );
        case FZNConstraintType::SET_SUBSET:
            return std::make_shared<SetSubset>( vars, args );
        case FZNConstraintType::SET_SUBSET_REIF:
            return std::make_shared<SetSubsetReif>( vars, args );
        case FZNConstraintType::SET_SYMDIFF:
            return std::make_shared<SetSymDiff>( vars, args );
        case FZNConstraintType::SET_UNION:
            return std::make_shared<SetUnion>( vars, args );
      default:
        std::cout << "Constraint \"" << c_name << "\" not yet implemented.\n";
        std::cout << "Default action: skip this constraint.\n";
        return nullptr;
    }
  }//get_fzn_constraint
  
};


#endif
