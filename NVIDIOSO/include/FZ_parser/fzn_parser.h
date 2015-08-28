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
     * Utility function for replacing bool vars with 0..1 int vars.
     * @param line input string
     * @param string where "var bool" is replaced by "var 0..1".
     */
     std::string replace_bool_vars ( std::string line );
     
     /**
      * Utility function for replacing true/false values in FlatZinc 
      * with 1/0 respectively.
      * @param line input string
      * @param string where "true" and "false" are replaced by "1" and "0" respectively.
      */
      std::string replace_bool_vals ( std::string line );
      
      /**
      * Utility function for converting Boolean FlatZinc input into 1/0 input.
      * @param line input string
      * @param converted string for iNVIDIOSO input.
      */
      std::string replace_bool ( std::string line );
    
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
    //! Ask whether there are more info/aux arrays to get
    bool more_aux_arrays     () const override;
    
    //! Ask whether there are more variables to get
    bool more_variables      () const override;
  
    //! Ask whether there are more constraits to get
    bool more_constraints    () const override;
  
    //! Ask whether there are more search engines to get
    bool more_search_engines () const override;
  
  	//! Ask whether there are more constraints stores to get
    bool more_constraint_stores () const override;
    
  	/**
     * Get a "aux/info array" token.
     * @return token pointer to a "aux/info array" token.
     * @note the returned pointer is a unique_ptr.
     */
  	UTokenPtr get_aux_array        () override;
  	
    /**
     * Get a "variable" token.
     * @return token pointer to a "variable" token.
     * @note the returned pointer is a unique_ptr.
     */
    UTokenPtr get_variable         () override;
  
    /**
     * Get a "constraint" token.
     * @return token pointer to a "constraint" token.
     * @note the returned pointer is a unique_ptr.
     */
    UTokenPtr get_constraint       () override;
  
    /**
     * Get a "search_engine" token.
     * @return token pointer to a "search_engine" token.
     * @note the returned pointer is a unique_ptr.
     */
    UTokenPtr get_search_engine    () override;
  
  	/**
     * Get a "constraint store" token.
     * @return token pointer to a "constraint_store" token.
     * @note the returned pointer is a unique_ptr.
     */
    UTokenPtr get_constraint_store () override;
    
    //! Print info about the parser
    void print () const;
};


#endif
