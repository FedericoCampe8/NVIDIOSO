//
//  bool_2_int.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//
//  Constraint 
//  Domain consistency is used.
//

#ifndef __NVIDIOSO__bool_2_int__
#define __NVIDIOSO__bool_2_int__

#include "base_constraint.h"
#include "int_variable.h"

class Bool2Int : public BaseConstraint {  
public:
    /**
     * Basic constructor.
     * @note after this constructor the client should
     *       call the setup method to setup the variables
     *       and parameters needed by this constraint.
     */
    Bool2Int ( std::string& constraint_name );

    ~Bool2Int ();

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
