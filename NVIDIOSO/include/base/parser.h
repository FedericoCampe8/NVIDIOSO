//
//  parser.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 06/30/14.
//  Modified on 6/22/15.
//  Copyright (c) 2015 ___UDNMSU___. All rights reserved.
//
//  Interface for a Parser.
//  It defines a general interface for implementing a parser for
//  different CP models (e.g., MiniZinc, FlatZinc, GeCode, etc.).
//  @note: to create a new Parser some classes must be specialized.
//         In particular:
//         1) Parser       - defines how to parse
//         2) Tokenization - tokenizes the (input) description of the model
//         3) Token        - defines tokens w.r.t. the syntax of the modeling language
//

#ifndef NVIDIOSO_parser_h
#define NVIDIOSO_parser_h

#include "globals.h"
#include "tokenization.h"
#include "token.h"

//! Add other modeling languages here if needed
enum class ParserType {
  FLATZINC,
  OTHER
};

class Parser {  
protected:
    //! Tokenizer: it tokenizes lines read from the input file
    Tokenization * _tokenizer;
  
    //! Input stream (from file)
    std::ifstream  * _if_stream;
    std::string _input_path;
    std::string _dbg;

    // Member variables
    bool _open_file;
    bool _open_first_time;
    bool _more_tokens;
    bool _new_line;
    bool _failure;
  
    //! Number of lines read so far
    int _current_line;
  
    //! Delimiter to use to tokenize words
    std::string _delimiters;
  
    //! Positions in stream (file)
    std::streampos _curr_pos;
    
    //! Constructor
    Parser ();
    Parser ( std::string );
    
public:
    virtual ~Parser(); 
  
    //! Set input
    void set_input ( std::string );
  
    //! Add delimiter to tokenizer.
    void add_delimiter ( std::string );
  
    //! Get current (parsed) line
    int get_current_line ();
  
    //! Check whether the parser failed to parse the file
    bool is_failed () const;
    
    /**
     * Check if the internal status has more tokens to
     * give back to the client. 
     */
    virtual bool more_tokens ();
  
    /**
     * Open the file.
     * The file is open (if not already open) and the pointer is placed
     * on the last position read.
     * If the file is open for the first time,
     * the pointer is placed on the first position.
     */
    virtual void open ();
  
    /**
     * Close the file.
     * @note: alternating open() and close()
     * the client can decided how much text
     * has to be parsed.
     * For example, parse only the first n lines
     * of the text file.
     */
    virtual void close ();
  
    /**
     * Get next token.
     * This function returns a string corresponding
     * to the token parsed according to the internal
     * state of the object (i.e., pointer in the text file). 
     */
    virtual std::string get_next_token ();

    /**
     * Returns a token at a time from the set of
     * tokens currently stored in the parser.
     * This is equivalent to call
     * get_variable ();
     * get_constraint ();
     * get_search_engine ();
     * until no tokens are available.
     * @return a (unique_ptr) pointer to the current token
     *         read from input
     * @note if no token can be read, it returns a null,
     *       empty object.
     */
    virtual UTokenPtr get_next_content ();

    /**
     * Parses the file.
     * It fills the internal state with tokens
     * created by reading the model.
     * @return True if parsed succeeded,
     *         False otherwise.
     */
    virtual bool parse () = 0;
    
    /**
     * Get methods:
     * more tokens of the same related type (i.e., variables,
     * constraints, and search engine).
     * These methods should be used together with the 
     * "get" methods.
     */
    virtual bool more_variables      () const = 0;
    virtual bool more_constraints    () const = 0;
    virtual bool more_search_engines () const = 0;
  
    /**
     * Get methods:
     * get variables, constraints, and the search engine.
     * They increment the counter of available tokens.
     * The tokens are returned in order w.r.t. their
     * variables.
     * @return return a unique_ptr
     */
    virtual UTokenPtr get_variable      () = 0;
    virtual UTokenPtr get_constraint    () = 0;
    virtual UTokenPtr get_search_engine () = 0;
  
    //! Print info
    virtual void print () const = 0;
};


#endif
