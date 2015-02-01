//
//  int_lt.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 13/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Constraint X #< Y.
//

#ifndef __NVIDIOSO__int_lt__
#define __NVIDIOSO__int_lt__

#include "fzn_constraint.h"
#include "int_variable.h"

class IntLt : public FZNConstraint {
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
  IntLt ();
  
  /**
   * Basic constructor.
   * @note this constructor implicitly calls the setup
   *       method to setup variables and arguments for
   *       this constraint.
   */
  IntLt ( std::vector<VariablePtr> vars, std::vector<std::string> args );
  
  /**
   * Basic constructor: it checks if x != y.
   * @param x an integer value.
   * @param y an integer value.
   */
  IntLt ( int x, int y );
  
  /**
   * Constructor.
   * @param x (pointer to) a FD variable.
   * @param y an integer value.
   * @note It subtracts the value y from
   *       the domain of the variable x if
   *       x has a domain defined on integers.
   */
  IntLt ( IntVariablePtr x, int y );
  
  /**
   * Constructor.
   * @param x an integer value.
   * @param y (pointer to) a FD variable.
   * @note It subtracts the value x from
   *       the domain of the variable y if
   *       y has a domain defined on integers.
   */
  IntLt ( int x, IntVariablePtr y );
  
  /**
   * Constructor.
   * @param x (pointer to) a FD variable.
   * @param y (pointer to) a FD variable.
   */
  IntLt ( IntVariablePtr x, IntVariablePtr y );
  
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

#endif /* defined(__NVIDIOSO__int_lt__) */
