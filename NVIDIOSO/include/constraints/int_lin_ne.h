//
//  int_lin_ne.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 11/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Constraint Sum_{i \in 1..n}: as[i].bs[i] #\= c,
//  ehre n is the common length of as and bs.
//  Domain consistency is used.
//

#ifndef __NVIDIOSO__int_lin_ne__
#define __NVIDIOSO__int_lin_ne__

#include "fzn_constraint.h"
#include "int_variable.h"

class IntLinNe : public FZNConstraint {
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
  
  /**
   * States whether at least one variable is ground.
   * @return true if at least one variables in the scope is assigned,
   *         false otherwise.
   */
  virtual bool at_least_one_ground ();
  
  /**
   * States whether there is only one non ground variable.
   * @return true if there is only one variable in the scope which has not
   *         been assigned yet, false otherwise.
   */
  virtual bool only_one_not_ground ();
  
public:
  /**
   * Basic constructor.
   * @note after this constructor the client should
   *       call the setup method to setup the variables
   *       and parameters needed by this constraint.
   */
  IntLinNe ();
  
  /**
   * Basic constructor.
   * @note this constructor implicitly calls the setup
   *       method to setup variables and arguments for
   *       this constraint.
   */
  IntLinNe ( std::vector<VariablePtr> vars, std::vector<std::string> args );
  
  ~IntLinNe();
  
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

#endif /* defined(__NVIDIOSO__int_lin_ne__) */
