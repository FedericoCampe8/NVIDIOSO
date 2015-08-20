//
//  int_lin_eq.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 20/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Constraint Sum_{i \in 1..n}: as[i].bs[i] #= c,
//  where n is the common length of as and bs.
//
#ifndef __NVIDIOSO__int_lin_eq__
#define __NVIDIOSO__int_lin_eq__

#include "base_constraint.h"
#include "int_variable.h"

class IntLinEq : public BaseConstraint {
private:
  //! array [int] of int
  std::vector<int> _as;
  
  //! array [int] of var int
  std::vector<IntVariablePtr> _bs;
  
  //! int
  int _c;
  
  /**
   * States whether all the variables in _bs are ground.
   * @return true if all variables in the scope are assigned,
   *         false otherwise.
   */
  virtual bool all_ground ();
  
public:
  /**
   * Basic constructor.
   * @note after this constructor the client should
   *       call the setup method to setup the variables
   *       and parameters needed by this constraint.
   */
  IntLinEq ( std::string& constraint_name );
  
  ~IntLinEq();
  
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


#endif /* defined(__NVIDIOSO__int_lin_eq__) */
