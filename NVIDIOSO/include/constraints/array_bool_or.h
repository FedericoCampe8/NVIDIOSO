//
//  array_bool_or.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//
//  Constraint 
//  Domain consistency is used.
//

#ifndef __NVIDIOSO__array_bool_or__
#define __NVIDIOSO__array_bool_or__

#include "fzn_constraint.h"
#include "int_variable.h"

class ArrayBoolOr : public FZNConstraint {  
public:
    /**
     * Basic constructor.
     * @note after this constructor the client should
     *       call the setup method to setup the variables
     *       and parameters needed by this constraint.
     */
    ArrayBoolOr ();

    ~ArrayBoolOr ();

    /**
     * Basic constructor.
     * @note this constructor implicitly calls the setup
     *       method to setup variables and arguments for
     *       this constraint.
     */
    ArrayBoolOr ( std::vector<VariablePtr> vars, std::vector<std::string> args );

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
  
    //! It checks if
    bool satisfied () override;
  
    //! Prints the semantic of this constraint
    void print_semantic () const override;
};

#endif
