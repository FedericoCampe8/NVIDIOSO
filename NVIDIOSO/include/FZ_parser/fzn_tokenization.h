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
    
    /**
     * FlatZinc specific keywords:
     * <string_1, string_2>
     * where
     * string_2 is a keyword from FlatZinc syntax, while
     * string_1 is a keyword (not necessarly different from string_2) used
     * within this parser to refer to string_2.
     * @note a map is used here for two reasons:
     *       1) future code extensions
     *       2) compatibility with older gcc compilers
     */
    std::unordered_map < std::string, std::string > _fzn_keywords;

    //! Set new keywork mapping lookup table
    void set_flatzinc_map ( std::unordered_map < std::string, std::string >& m );
    
    //! Add new keywork to lookup table
    void add_flatzinc_word ( std::string key, std::string value );
  
    //! Specialized method
    UTokenPtr analyze_token (); // override
    
    // Some methods used to analyze FlatZinc predicates
  
    /**
     * Analyze the token corresponding to an 
     * array of variables.
     * @return returns the corresponding (pointer to) token
     *         initialized with the information read 
     *         from the model.
     */
    UTokenPtr analyze_token_arr ();
  
    /**
     * Analyze the token corresponding to a constraint
     * @return returns the (pointer to) token
     *         initialized with the information read
     *         from the model.
     */
    UTokenPtr analyze_token_con ();
  
    /**
     * Analyze the token corresponding to a solution predicate.
     * @return returns the (pointer to) token
     *         initialized with the information read
     *         from the model.
     */
    UTokenPtr analyze_token_sol ();
  
    /**
     * Analyze the token corresponding to a variable
     * @return returns the (pointer to) token
     *         initialized with the information read
     *         from the model.
     */
    UTokenPtr analyze_token_var ();
  
    /**
     * Analyze the token corresponding to a LNS predicate
     * @return returns the (pointer to) token
     *         initialized with the information read
     *         from the model.
     */
    UTokenPtr analyze_token_lns ();
  
public:
    FZNTokenization  ();
    ~FZNTokenization ();
  
  	/**
  	 * This function is used for testing.
  	 * Sets the current line to tokenize.
  	 * @param: string of at most 250 chars to tokenize.
  	 */
  	 void set_internal_state ( std::string str );
  	 
    /** 
     * Specialized method:
     * It actually gets the right token
     * according to the FlatZinc format.
     * Analysis is perfomed on "_c_token".
     */
    UTokenPtr get_token (); // override
};


#endif
