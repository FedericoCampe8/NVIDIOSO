//
//  alldifferent.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 31/07/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class specializes global constraints for alldifferent.
//  There are three different propagation algorithms (i.e., three propagators)
//  that can be used:
//  1 - Eliminate singletons by naive value propagation.
//  2 - Bounds consistent alldifferent propagator.
//      Algorithm taken from:
//      	A. Lopez-Ortiz, C.-G. Quimper, J. Tromp, and P. van Beek.
//        	A fast and simple algorithm for bounds consistency of the
//        	alldifferent constraint. IJCAI-2003.
//
//      This implementation uses the code that is provided by Peter Van Beek:
//			http://ai.uwaterloo.ca/~vanbeek/software/software.html
//  3 - Domain consistent distinct propagator.
//		The algorithm is taken from:
// 			Jean-Charles RÈgin, A filtering algorithm for constraints
//		    of difference in CSPs, Proceedings of the Twelfth National
//			Conference on Artificial Intelligence, pages 362--367.
//			Seattle, WA, USA, 1994.
//


#ifndef __NVIDIOSO__alldifferent__
#define __NVIDIOSO__alldifferent__

#include "global_constraint.h"

class Alldifferent : public GlobalConstraint {
protected:
	
public:
  Alldifferent ( std::string name );
  
  virtual ~Alldifferent ();
  
  /**
   * It sets the variables and the arguments for this constraint.
   * @param vars a vector of pointers to the variables in the
   *        constraint's scope.
   * @param args a vector of strings representing the auxiliary
   *        arguments needed by the constraint in order to ensure
   *        consistency.
   */
  virtual void setup ( std::vector<VariablePtr> vars, std::vector<std::string> args );
  
  /**
   * It is a (most probably incomplete) consistency function which
   * removes the values from variable domains. Only values which
   * do not have any support in a solution space are removed.
   */
  void consistency () override;
  
  /**
   * It checks if the constraint is satisfied.
   * @return true if the constraint if for certain satisfied,
   *         false otherwise.
   * @note If this function is incorrectly implementd,
   *       a constraint may not be satisfied in a solution.
   */
  bool satisfied () override;
  
  //! Prints info.
  void print () const override;
  
  //! Prints the semantic of this constraint.
  void print_semantic () const override;
};

#endif /* defined(__NVIDIOSO__global_constraint__) */
