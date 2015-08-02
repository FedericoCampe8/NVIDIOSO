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
     * Lookup table for tokens with <key, value> pair as follows
     * <pos, ptr>
     * where pos is the sequence number of the
     * token pointed by ptr.
     */
    std::map < size_t, UTokenPtr > _map_tokens;
    
    /**
     * Stores the tokens both in the lookup table 
     * and in the map of tokens.
     * Moreover, it defines new tokens for arrays of variables.
     * @param UTokenPtr a unique_ptr to the token.
     * @note this is a meta-declaration of tokens.
     *       The tokenizer returns a token corresponding to
     *       an array of variables.
     *       The parser split the array creating new tokens.
     * @note store_token claim ownership of the pointer.
     *       The above means that UTokenPtr has to be passed
     *       by move (std::move) on an actual l-value,
     *       i.e., a named variable, and it cannot be called
     *       with a temprary (r) value.
     */
    void store_token ( UTokenPtr );
  
  	//! Ask whether there are more constraits to get
    bool more_base_constraints    () const;
    
    //! Ask whether there are more constraits to get
    bool more_global_constraints  () const;
    
public:
  
    FZNParser ();
    FZNParser ( std::string ifile );

    //! Parses the file filling the internal state of the parser
    bool parse (); //override
    
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
     * @note the returned pointer is a unique_ptr.
     */
    UTokenPtr get_variable      ();
  
    /**
     * Get a "constraint" token.
     * @return token pointer to a "constraint" token.
     * @note the returned pointer is a unique_ptr.
     */
    UTokenPtr get_constraint    ();
  
    /**
     * Get a "search_engine" token.
     * @return token pointer to a "search_engine" token.
     * @note the returned pointer is a unique_ptr.
     */
    UTokenPtr get_search_engine ();
  
    //! Print info about the parser
    void print () const;
};


#endif
