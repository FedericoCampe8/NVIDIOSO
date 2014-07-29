//
//  int_ne.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 29/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Constraints X #\= Y.
//  Domain consistency is used.
//

#ifndef __NVIDIOSO__int_ne__
#define __NVIDIOSO__int_ne__

#include "fzn_constraint.h"
#include "int_variable.h"

class IntNe : public FZNConstraint {
private:
  IntVariablePtr _var_x = nullptr;
  IntVariablePtr _var_y = nullptr;
  
public:
  /**
   * Basic constructor: it checks if x != y.
   * @param x an integer value.
   * @param y an integer value.
   */
  IntNe ( int x, int y );
  
  /**
   * Constructor.
   * @param x (pointer to) a FD variable.
   * @param y an integer value.
   * @note It subtracts the value y from
   *       the domain of the variable x if
   *       x has a domain defined on integers.
   */
  IntNe ( IntVariablePtr x, int y );
  
  /**
   * Constructor.
   * @param x an integer value.
   * @param y (pointer to) a FD variable.
   * @note It subtracts the value x from
   *       the domain of the variable y if
   *       y has a domain defined on integers.
   */
  IntNe ( int x, IntVariablePtr y );
  
  /**
   * Constructor.
   * @param x (pointer to) a FD variable.
   * @param y (pointer to) a FD variable.
   */
  IntNe ( IntVariablePtr x, IntVariablePtr y );
  
  /**
   * It returns the vector of (shared) pointers
   * of all the variables involved in a
   * given constraint (i.e., its scope).
   */
  const std::vector<VariablePtr> scope () const;
  
  //! It performs domain consistency
  void consistency () override;
  
  //! It checks if x != y
  bool satisfied () override;
  
  //! Prints the semantic of this constraint
  void print_semantic () const override;
};

#endif /* defined(__NVIDIOSO__int_ne__) */
