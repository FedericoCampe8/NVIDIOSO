//
//  int_times.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Modified by Luca Foschiani on 08/18/15 (foschiani01@gmail.com).
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  Constraint x * y = z
//  Domain consistency is used.
//

#ifndef __NVIDIOSO__int_times__
#define __NVIDIOSO__int_times__

#include "fzn_constraint.h"
#include "int_variable.h"

class IntTimes : public FZNConstraint {
private:
    IntVariablePtr _var_x = nullptr;
    IntVariablePtr _var_y = nullptr;
    IntVariablePtr _var_z = nullptr;

    std::pair<int,int> mul_bounds ( int d1, int d2, int e1, int e2 );
    std::pair<int,int> div_bounds ( int d1, int d2, int e1, int e2 );

public:
    /**
     * Basic constructor.
     * @note after this constructor the client should
     *       call the setup method to setup the variables
     *       and parameters needed by this constraint.
     */
    IntTimes ();

    /**
     * Basic constructor.
     * @note this constructor implicitly calls the setup
     *       method to setup variables and arguments for
     *       this constraint.
     */
    IntTimes ( std::vector<VariablePtr> vars, std::vector<std::string> args );

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
