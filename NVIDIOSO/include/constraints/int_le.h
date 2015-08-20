//
//  int_le.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 13/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Constraint X #<= Y.
//


#ifndef __NVIDIOSO__int_le__
#define __NVIDIOSO__int_le__

#include "base_constraint.h"
#include "int_variable.h"

class IntLe : public BaseConstraint {
private:
  IntVariablePtr _var_x = nullptr;
  IntVariablePtr _var_y = nullptr;
  
public:
  /**
   * Basic constructor.
   * @note after this constructor the client should
   *       call the setup method to setup the variables
   *       and parameters needed by this constraint.
   */
  IntLe ( std::string& constraint_name );
  
  //! Setup method, see fzn_constraint.h
  void setup ( std::vector<VariablePtr> vars, std::vector<std::string> args ) override;
  
  /**
   * It returns the vector of (shared) pointers
   * of all the variables involved in a
   * given constraint (i.e., its scope).
   */
  const std::vector<VariablePtr> scope () const override;
  
  //! It performs domain consistency
  void consistency () override;
  
  //! It checks if x != y
  bool satisfied () override;
  
  //! Prints the semantic of this constraint
  void print_semantic () const override;
};


#endif /* defined(__NVIDIOSO__int_le__) */
