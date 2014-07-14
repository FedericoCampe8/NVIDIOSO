//
//  solver.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 27/06/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class provides the interface for a general constraint solver

#ifndef NVIDIOSO_solver_h
#define NVIDIOSO_solver_h

class Solver {
public:
  Solver() {};
  ~Solver() {};
  
  virtual void run() = 0;
};


#endif
