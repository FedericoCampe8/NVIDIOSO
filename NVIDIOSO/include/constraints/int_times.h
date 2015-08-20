//
//  int_times.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Modified by Luca Foschiani on 08/18/15 (foschiani01@gmail.com).
//  Copyright (c) 2014-2015 Federico Campeotto. 
//	All rights reserved.
//
//  Constraint x * y = z
//  Domain consistency is used.
//

#ifndef __NVIDIOSO__int_times__
#define __NVIDIOSO__int_times__

#include "base_constraint.h"
#include "int_variable.h"

class IntTimes : public BaseConstraint {
private:
    IntVariablePtr _var_x = nullptr;
    IntVariablePtr _var_y = nullptr;
    IntVariablePtr _var_z = nullptr;

    /**
     * Returns the bounds of the domain
     * [d1..d2]*[e1..e2]
     */
    std::pair<int,int> mul_bounds ( int d1, int d2, int e1, int e2 );

    /**
     * Returns the bounds of the domain
     * [d1..d2]/[e1..e2]
     * @todo implement rounding for integer division:
     *       ceil for the lower bound
     *       floor for the upper bound
     *       this would make the domains smaller in many cases.
     */
    std::pair<int,int> div_bounds ( int d1, int d2, int e1, int e2 );

public:
    /**
     * Basic constructor.
     * @note after this constructor the client should
     *       call the setup method to setup the variables
     *       and parameters needed by this constraint.
     */
    IntTimes ( std::string& constraint_name );

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
