//
//  fzn_tokenization.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 04/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This tokenizer is specialized for FlatZinc file
//  It specializes tokenization.
//  The tokenizer splits a string in tokens and manages them
//  in order to create the appropriate tokens representing
//  objects in the model.
//  For example, given the string:
//  "array [1..2] of var set of 1..3: a;"
//  the tokenizer splits it into substrings and
//  performs different actions according to the substring:
//  - "array"  : instantiates a token_arr;
//  - "[1..2]" : sets the size of the array;
//  - "set"    : defines the type of domain;
//  - "1..3"   : defines the type of set-domain;
//  - "a"      : defines the name (id) of the array.
//

#ifndef NVIDIOSO_fzn_tokenization_h
#define NVIDIOSO_fzn_tokenization_h

#include "tokenization.h"

class FZNTokenization : public Tokenization {
private:
  // FlatZinc specific keywords
  const std::string ARR_TOKEN = "array";
  const std::string BOO_TOKEN = "bool";
  const std::string CON_TOKEN = "constraint";
  const std::string FAL_TOKEN = "false";
  const std::string FLO_TOKEN = "float";
  const std::string INT_TOKEN = "int";
  const std::string LNS_TOKEN = "lns";
  const std::string MIN_TOKEN = "minimize";
  const std::string MAX_TOKEN = "maximize";
  const std::string OAR_TOKEN = "output_array";
  const std::string OBJ_TOKEN = "obj_var";
  const std::string OUT_TOKEN = "output";
  const std::string RAN_TOKEN = "..";
  const std::string SAT_TOKEN = "satisfy";
  const std::string SET_TOKEN = "set";
  const std::string SHO_TOKEN = "show";
  const std::string SOL_TOKEN = "solve";
  const std::string VAR_TOKEN = "var";
  const std::string VII_TOKEN = ":: var_is_introduced";
  
  //! Specialized method
  TokenPtr analyze_token ();
  
  // Some methods used to analyze FlatZinc predicates
  
  /**
   * Analyze the token corresponding to an 
   * array of variables.
   * @return returns the corresponding (pointer to) token
   *         initialized with the information read 
   *         from the model.
   */
  TokenPtr analyze_token_arr ();
  
  /**
   * Analyze the token corresponding to a constraint
   * @return returns the (pointer to) token
   *         initialized with the information read
   *         from the model.
   */
  TokenPtr analyze_token_con ();
  
  /**
   * Analyze the token corresponding to a solution predicate.
   * @return returns the (pointer to) token
   *         initialized with the information read
   *         from the model.
   */
  TokenPtr analyze_token_sol ();
  
  /**
   * Analyze the token corresponding to a variable
   * @return returns the (pointer to) token
   *         initialized with the information read
   *         from the model.
   */
  TokenPtr analyze_token_var ();
  
  /**
   * Analyze the token corresponding to a LNS predicate
   * @return returns the (pointer to) token
   *         initialized with the information read
   *         from the model.
   */
  TokenPtr analyze_token_lns ();
  
public:
  
  FZNTokenization  ();
  ~FZNTokenization ();
  
  /** 
   * Specialized method:
   * It actually gets the right token
   * according to the FlatZinc format.
   * Analysis is perfomed on "_c_token".
   */
  TokenPtr get_token ();
};


#endif
