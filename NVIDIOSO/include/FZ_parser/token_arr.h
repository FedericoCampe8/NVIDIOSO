//
//  token_arr.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 05/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#ifndef NVIDIOSO_token_arr_h
#define NVIDIOSO_token_arr_h

#include "token_var.h"

class TokenArr : public TokenVar {
private:
  // Info about the array of variables
  int  _size;
  int  _array_lwb;
  int  _array_upb;
  
  // Global idx of the variables within the array of vars
  int  _lw_var_glb_idx;
  int  _up_var_glb_idx;
  bool _output_arr;
 
public:
  TokenArr ();
  
  // Get/set methods
  void set_size_arr ( int );
  int  get_size_arr () const;
  
  /**
   * Array set and info.
   * For example, array [1..30] of ...
   * get_lw_bound -> 1
   * get_lw_bound -> 30
   * It sets the bounds of the array.
   * @param lw lower bound
   * @param up upper bound
   */
  void set_array_bounds ( int lw, int up );
  int  get_lw_bound () const;
  int  get_up_bound () const;
  
  /**
   * Variables (idx) within the array.
   * The index is given w.r.t. the global index of 
   * parsed tokens so far.
   * @return the lower idx of variable within the array
   */
  int get_lower_var () const;
  
  /**
   * Variables (idx) within the array.
   * The index is given w.r.t. the global index of
   * parsed tokens so far.
   * @return the higher idx of variable within the array
   */
  int get_upper_var () const;
  
  /**
   * Check whether a given variable (idx) is 
   * indexed by the array (i.e., is whithin  the array.
   * @note: check is performed w.r.t. both the variable
   *        string identifier (e.g., a[i])
   *        and its global id.
   * @param var the variable to check membership
   * @return true if var is in the current array, false otherwise
   */
  bool is_var_in ( int var ) const;
  bool is_var_in ( std::string ) const;
  
  //! Identifies the current variable array as a support variable array
  void set_output_arr ();
  bool is_output_arr  () const;
  
  //! Print info methods
  void print () const;
};

#endif