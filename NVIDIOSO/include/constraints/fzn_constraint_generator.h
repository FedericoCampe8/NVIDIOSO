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

#include "int_ne.h"
#include "int_lin_ne.h"
#include "int_le.h"
#include "int_lt.h"
#include "int_eq.h"
#include "int_lin_eq.h"

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
    
    switch ( c_type ) {
      case FZNConstraintType::INT_NE:
        return std::make_shared<IntNe>( vars, args );
      case FZNConstraintType::INT_LIN_NE:
        return std::make_shared<IntLinNe>( vars, args );
      case FZNConstraintType::INT_LE:
        return std::make_shared<IntLe>( vars, args );
      case FZNConstraintType::INT_LT:
        return std::make_shared<IntLt>( vars, args );
      case FZNConstraintType::INT_EQ:
        return std::make_shared<IntEq>( vars, args );
      case FZNConstraintType::INT_LIN_EQ:
        return std::make_shared<IntLinEq>( vars, args );
      default:
        std::cout << "Constraint \"" << c_name << "\" not yet implemented.\n";
        std::cout << "Default action: skip this constraint.\n";
        return nullptr;
    }
  }//get_fzn_constraint
  
};


#endif
