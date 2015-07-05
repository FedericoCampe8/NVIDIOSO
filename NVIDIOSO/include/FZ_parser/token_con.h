//
//  token_con.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class specializes a token for constraints.
//  @note: this class represents the minium set of information
//         needed to identify the constraint.
//         Link between variables and constraints shall be performed
//         in the appropriate classes.
//

#ifndef NVIDIOSO_token_con_h
#define NVIDIOSO_token_con_h

#include "token.h"

class TokenCon : public Token {
protected:
  
  //! Info about the constraint
  std::string _con_id;
  
  //! Parameters involved in the constraint
  std::vector < std::string > _exprs;
  
public:
  
  TokenCon ();
  
  bool set_token ( std::string& token_string ) override;
  
  //! Set method constraint id (i.e., constraint's name).
  void set_con_id ( std::string );
  
  //! Get the string representing the constraint's name.
  std::string get_con_id () const;
  
  /**
   * Add expression (parameters) to the 
   * token that identifies the parsed constraint.
   * For example,
   * constraint int_ne(magic[1], magic[2])
   * expression = "magic[1]" and "magic[2]"
   * @param str string representing the expression.
   */
  void add_expr ( std::string str );
  
  //! Get the number of parameters needed by the constraint
  int  get_num_expr () const;
  
  /**
   * Get the string represeting the ith expression
   * that defines the constraint.
   * @param idx index of the expression to return
   * @return return the idx^th expression
   */
  std::string get_expr ( int ) const;
  
  /**
   * Return an array containing all the (string) expressions
   * that define the current constraint.
   * @return a vector of strings representing the expressions
   *         defining this constraint.
   */
  const std::vector<std::string> get_expr_array ();
  
  /**
   * Return an array containing all the (string) elements 
   * of each expression that define the current constraint.
   * @return a vector of strings representing the elements of
   *         each expression that defines this constraint.
   * @note the strings in output preserves the order as found in
   *       the original string token.
   */
  const std::vector<std::string> get_expr_elements_array ();
  
  /**
   * Return an array containing all the (string) "variable" elements
   * of each expression that define the current constraint.
   * @return a vector of strings representing the "variable" elements of
   *         each expression that defines this constraint.
   * @note the strings in output preserves the order as found in
   *       the original string token.
   */
  const std::vector<std::string> get_expr_var_elements_array ();
  
  /**
   * Return an array containing all the (string) "non variable" elements
   * of each expression that define the current constraint.
   * @return a vector of strings representing the "non variable" elements of
   *         each expression that defines this constraint.
   * @note the strings in output preserves the order as found in
   *       the original string token.
   */
  const std::vector<std::string> get_expr_not_var_elements_array ();
  
  //! Print info methods
  virtual void print () const;
};

#endif
