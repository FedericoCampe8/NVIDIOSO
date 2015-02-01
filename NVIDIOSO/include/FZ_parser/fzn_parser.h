//
//  fzn_parser.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 30/06/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class implements a Parser for FlatZinc models.
//  It uses a Tokenizer to tokenize lines. It keeps an internal
//  state in order to link constraints to parsed variables defined
//  somewhere in the file.
//  @note first the parser fills the lookup table of tokens, then
//        the client asks for some tokens, i.e., the order in which
//        the client asks for tokens may not reflect the order of
//        the lines parsed in the input file.
//

#ifndef NVIDIOSO_fzn_parser_h
#define NVIDIOSO_fzn_parser_h

#include "globals.h"
#include "parser.h"

class FZNParser : public Parser {
private:
  //! Total number of tokens parsed so far
  size_t _num_tokens;
  
  /**
   * Lookup table for tokens:
   * @note using this lookup table the client can store
   *       tokens related to variables, constraints and
   *       search strategies in any order.
   * @params TokenType as key -> vector of tokens id
   *         as stored in the _map_tokens map.
   */
  std::map < TokenType, std::vector< size_t > > _lookup_token_table;
  
  /**
   * Stores the tokens both in the lookup table 
   * and in the map of tokens.
   * Moreover, it defines new tokens for arrays of variables.
   * @note this is a meta-declaration of tokens.
   *       The tokenizer returns a token corresponding to
   *       an array of variables.
   *       The parser split the array creating new tokens.
   */
  void store_token ( TokenPtr );
  
public:
  
  FZNParser ();
  FZNParser ( std::string ifile );
  
  // Get tokens w.r.t. their type
  //! Ask whether there are more variables to get
  bool more_variables      () const;
  
  //! Ask whether there are more constraits to get
  bool more_constraints    () const;
  
  //! Ask whether there are more search engines to get
  bool more_search_engines () const;
  
  /**
   * Get a "variable" token.
   * @return token pointer to a "variable" token.
   */
  TokenPtr get_variable      ();
  
  /**
   * Get a "constraint" token.
   * @return token pointer to a "constraint" token.
   */
  TokenPtr get_constraint    ();
  
  /**
   * Get a "search_engine" token.
   * @return token pointer to a "search_engine" token.
   */
  TokenPtr get_search_engine ();
  
  //! Get next (pointer to) token (i.e., FlatZinc element)
  TokenPtr get_next_content ();
  
  //! Print info about the parser
  void print () const;
};


#endif
