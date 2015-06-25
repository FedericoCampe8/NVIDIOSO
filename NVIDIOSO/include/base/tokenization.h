//
//  tokenization.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 01/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Tokenization class implements a tokenizer.
//  It implements the heuristics used to tokenize the
//  string read by the parser.
//  It keeps an internal state and for each line it returns
//  the correspondend object of type "Token".
//

#ifndef NVIDIOSO_tokenization_h
#define NVIDIOSO_tokenization_h

#include "globals.h"
#include "token.h"

class Tokenization {
protected:
    std::string _dbg;
  
    std::string DELIMITERS;
    std::string WHITESPACE;
  
    std::string _comment_lines;
  
    // Variables for identifying when a new line has been found
    bool _new_line;
    bool _need_line;
  
    // Other info
    bool _failed;
  
    // Token returned by strtok
    char * _c_token;
    
    //! Parsed line
    char * _parsed_line;
  
    // Other useful methods to tokenize a string
  
    //! It states whether the current char has to be skipped or not
    virtual bool avoid_char ( char );

    //! It states whether _c_token or a line must be skipped or not
    virtual bool skip_line ();
    virtual bool skip_line ( std::string );
  
    /**
     * It states whether a new line has been found.
     * Different heuristics may be used here.
     */
    virtual bool set_new_line ();
  
    /**
     * It "clears" the text line by removing
     * possible initial white spaces from line.
     * Different heuristics may be used here.
     */
    virtual void clear_line ();
  
    /**
     * Analyze token:
     * this function acts like a filter.
     * It analyzes _c_token and returns
     * a string corresponding to the token cleaned from
     * useless chars. 
     */
    virtual UTokenPtr analyze_token () = 0;
   
public:
    Tokenization  ();
    virtual ~Tokenization ();
  
    // Set/Add string delimiters
    void add_delimiter    ( std::string );
    void set_delimiter    ( std::string );
    void add_white_spaces ( std::string );
    void set_white_spaces ( std::string );
  
    /**
     * Prepare a new tokenizer (i.e., string for strtok).
     * @param line the string to tokenize.
     */
    void set_new_tokenizer ( std::string line );
  
    //! Informs whether a new line has been found
    bool find_new_line ();
  
    //! Check whether the tokenizer has failed
    bool is_failed () const;
  
    //! Asks whether the tokenizer has finished all the tokens
    bool need_line ();
  
    //! Set preferences
    void add_comment_symb ( char );
    void add_comment_symb ( std::string );
  
    //! Get the string correspondent to the (filtered) token
    virtual UTokenPtr get_token () = 0;
};


#endif


