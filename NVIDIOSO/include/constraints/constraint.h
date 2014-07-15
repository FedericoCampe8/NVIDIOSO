//
//  constraint.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class represents an interface for all constraints.
//  It defines how to construct a constraint, impose, check satisiability,
//  enforce consistency, etc.
//  Specific implementations based on hardware/software capabilities (e.g., CUDA)
//  should derive from this class.
//

#ifndef NVIDIOSO_constraint_h
#define NVIDIOSO_constraint_h

#include "globals.h"

class Constraint;
typedef std::shared_ptr<Constraint> ConstraintPtr;

class Constraint {
  
public:
  Constraint ();
  virtual ~Constraint ();
  
  
};


#endif
