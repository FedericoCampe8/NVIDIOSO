//
//  heuristic.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class defines the interface for the heuristic to be used
//  during search. Implementing this interface allows the client to
//  define different heuristics by defining different methods for
//  selecting the next variable to label as well as the next value
//  to assign to the selected variable.
//

#ifndef __NVIDIOSO__heuristic__
#define __NVIDIOSO__heuristic__

#include "globals.h"
#include "variable.h"

class Heuristic;
typedef std::shared_ptr< Heuristic > HeuristicPtr;

class Heuristic {
protected:
  //! Debug info
  std::string _dbg;
  
  //! Current index used to select the next choice variable
  int _current_index;
  
  Heuristic ();
  
public:
  virtual ~Heuristic ();
  
  /**
   * Return the current index (last index used) to
   * select the choice variable.
   */
  virtual int get_index () const;
  
  /**
   * Returns the variable which will represent
   * the next choice point (i.e., the next variable
   * to label). 
   * @param idx the position of the last variable which has 
   *        been returned by this heuristic and which has not been
   *        backtracked upon yet.
   * @return a reference to the variable to label in the next step 
   *         according to this heuristic. 
   *         nullptr is returned if all variables are assigned.
   */
  virtual Variable * get_choice_variable ( int idx ) = 0;
  
  /**
   * Returns a value which will represent the next choice point 
   * (i.e., the next value to assign to the variable selected
   *  by this huristic).
   * @return the value used in the choice point (value)
   * @note this value is an integer value. 
   *       If variables are not defined on integer values 
   *       (e.g., float vars), this method should either 
   *       be implemented consistently or never used.
   */
  virtual int get_choice_value () = 0;
  
  //! Print info about heuristic
  virtual void print () const = 0;
};


#endif /* defined(__NVIDIOSO__heuristic__) */
