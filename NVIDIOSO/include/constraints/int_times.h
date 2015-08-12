//
//  int_times.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  Domain consistency is used.
//  Constraint x * y = z
//  Domain consistency is used.
//  @note For this constraint we must consider the following situations:
//        1 - Three parameters ground: no problem, just sat check, e.g., 2 * 3 = 6;
//        2 - Two parameters ground: no such case, this is converted into equality constraint
//            during parsing, e.g., x * 3 = 5 -> int_eq (x, 1);
//        3 - One parameter ground: no consistency is performed, e.g., x * 3 = z or x * y = 5;
//        4 - Zero parameters ground: consistency algorithms are used, e.g., x * y = z.            
//

#ifndef __NVIDIOSO__int_times__
#define __NVIDIOSO__int_times__

#include "fzn_constraint.h"
#include "int_variable.h"

class IntTimes : public FZNConstraint {  
public:
    /**
     * Basic constructor.
     * @note after this constructor the client should
     *       call the setup method to setup the variables
     *       and parameters needed by this constraint.
     */
    IntTimes ();

    ~IntTimes ();

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
