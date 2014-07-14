//
//  cp_constraint_store.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class represents the interface for a Constraint Store.
//  It holds the information about the state of the process related
//  to the constraints and the constraints to be propagated.
//  Any Constraint Store specific for an application/hardware (e.g., CUDA)
//  must implement the method specified below.
//

#ifndef NVIDIOSO_constraint_store_h
#define NVIDIOSO_constraint_store_h

#include "globals.h"
#include "constraint.h"

class ConstraintStore {
  
public:
  ConstraintStore ();
  virtual ~ConstraintStore ();
  
};



#endif
