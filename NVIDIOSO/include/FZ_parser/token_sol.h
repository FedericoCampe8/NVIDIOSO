//
//  token_sol.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Token for representing the "solve" predicate.
//

#ifndef NVIDIOSO_token_sol_h
#define NVIDIOSO_token_sol_h

#include "token.h"

class TokenSol : public Token {
protected:
  std::string _var_goal;
  std::string _solve_goal;
  std::string _search_choice;
  std::string _label_choice;
  std::string _variable_choice;
  std::string _assignment_choice;
  std::string _strategy_choice;
  
  /**
   * Vector of strings corresponding to the variables
   * to label during the search phase.
   */
  std::vector < std::string > _var_to_label;
  
public:
  TokenSol ();
  
  //Get/Set methods
  void set_var_goal          ( std::string );
  void set_solve_goal        ( std::string );
  void set_solve_params      ( std::string );
  void set_label_choice      ( std::string );
  void set_search_choice     ( std::string );
  void set_variable_choice   ( std::string );
  void set_assignment_choice ( std::string );
  void set_strategy_choice   ( std::string );
  
  //! Set the (string) identifier of a variable to label.
  void set_var_to_label ( std::string );
  
  //! Var goal to optimize (if any).
  std::string get_var_goal          () const;
  
  //! Solve goal: satisfy, minimize, maximize.
  std::string get_solve_goal        () const;
  
  //! int_search, bool_search, set_search (if any).
  std::string get_search_choice     () const;
  
  //! Variables to be assigned (if any).
  std::string get_label_choice      () const;
  
  //! input_order, first_fail, etc, (if any).
  std::string get_variable_choice   () const;
  
  //! indomain_min, indomain_max, etc, (if any).
  std::string get_assignment_choice () const;
  
  //! complete, lns, etc, (if any).
  std::string get_strategy_choice   () const;
  
  /**
   * Number of variables to label if specified by the model.
   * @return the number of variables to label.
   */
  int num_var_to_label () const;
  
  /** 
   * Identifiers of the variables to label.
   * @return a vector of string identifiers of the variable
   *         to label during the search phase.
   */
  const std::vector< std::string > get_var_to_label () const;
  
  /**
   * Get the string corresponding to the ith variable to label.
   * @param idx the index of the variable to label.
   * @return the string identifier of the idx^th variable to label.
   */
  std::string get_var_to_label ( int idx ) const;
  
  //! Print info methods
  virtual void print () const;
};


#endif
