//
//  token_var.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 05/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class specializes a token for variables
//

#ifndef NVIDIOSO_token_var_h
#define NVIDIOSO_token_var_h

#include "token.h"

/**
 * @note: this domain types differ 
 * from the possible domains allowed by the system,
 * i.e., see domain.h.
 * The following types decouple the types read from the 
 * model from the actual types implemented by the system.
 */
enum class VarDomainType {
  BOOLEAN,
  FLOAT,
  INTEGER,
  RANGE,
  SET,
  SET_INT,
  SET_RANGE,
  OTHER
};

class TokenVar : public Token {
  
protected:
  // Info about the variable
  std::string _var_id;
  bool _objective_var;
  bool _support_var;
  VarDomainType _var_dom_type;
  
  // Aux for range and set domains
  int _lw_bound;
  int _up_bound;
  std::vector < std::vector <int> > _subset_domain;
  
  /**
   * Get a pair <x1, x2> from a
   * string of type "*x1..x2*".
   * @param str string to parse
   * @return a pair representing the range expressed with str
   */
  std::pair<int, int> get_range ( std::string str ) const;
  
  /**
   * Get a vector of elements from a
   * string of type "*{x1, x2, ...xk}*".
   * @param str string to parse
   * @return a pair representing the range expressed with str
   */
  std::vector<int> get_subset ( std::string str ) const;
  
public:
  TokenVar ();
  
  // Get/set methods
  /**
   * Set the (string) identifier of the
   * variable represented as a token.
   * The id is retrieved using the get_var_id() method.
   * @param str the string identifier of the variable.
   */
  void set_var_id ( std::string str );
  std::string get_var_id () const;
  
  //! Identifies the current variable as an objective variable
  void set_objective_var ();
  bool  is_objective_var () const;
  //! Identifies the current variable as a support variable
  void set_support_var ();
  bool  is_support_var () const;
  
  /**
   * Set the type of the current (token) variable.
   * @param vdt the variable domain type of type VarDomainType.
   */
  void set_var_dom_type ( VarDomainType vdt );
  VarDomainType get_var_dom_type () const;
  
  //! Specifies a boolean domain for the variable
  void set_boolean_domain ();
  //! Specifies a float domain for the variable
  void set_float_domain ();
  //! Specifies an integer domain for the variable
  void set_int_domain ();
  
  /**
   * Specifies a range domain for the variable
   * with a given a string of type "*x1..x2*".
   */
  void set_range_domain ( std::string str );
  
  /**
   * Specifies a range domain for the variable
   * with a given lower and upper bound.
   * @param lw lower bound
   * @param ub upper bound
   */
  void set_range_domain ( int lw, int ub );
  int  get_lw_bound_domain () const;
  int  get_up_bound_domain () const;
  
  /**
   * Call the right subset function, parsing
   * the string given in input.
   */
  void set_subset_domain ( std::string str );
  
  /**
   * Specifies a set of int domain.
   * @note set of int;
   */
  void set_subset_domain ();
  
  /**
   * Specifies a subsets of set domain for the variable
   * with the given vector of elements.
   * @param elems vector of elements
   * @note set of {x1, x2, ...xk}
   */
  void set_subset_domain ( const std::vector < int >& elems );
  
  /**
   * Specifies a subsets of set domain for the variable
   * with the given vector of elements.
   * @param elems vector of vectors of elements
   * @note set as {{x1, x2, ...xk}, ...}
   */
  void set_subset_domain ( const std::vector < std::vector <  int > >& elems );
   
  /**
   * Specifies a set of ints in range domain for the variable
   * with the given range.
   * @param range pair of int elements for range
   * @note set of x1..x2
   */
  void set_subset_domain ( const std::pair <int, int>& range );
  
  /**
   * Get the set of subsets of values for a var set type.
   * @return a vector of vectors of values representing the
   *         subsets of the var set type domain.
   */
  const std::vector < std::vector < int > > get_subset_domain ();
  
  //! Print info methods
  virtual void print () const;
};


#endif
