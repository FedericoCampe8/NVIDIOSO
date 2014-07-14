//
//  parser.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 30/06/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Interface for a Parser.
//  It defines a general interface for implementing a parser for
//  different CP models (e.g., MiniZinc, FlatZinc, GeCode, etc.).
//  @note: to create a new Parser some classes must be specialized.
//         In particular:
//         1) Parser - defines how to proceed in parsing
//         2) Tokenization - tokenizes the (input) description of the model
//         3) Token - defines tokens w.r.t. the modelling language to parse
//

#ifndef NVIDIOSO_parser_h
#define NVIDIOSO_parser_h

#include "globals.h"
#include "tokenization.h"
#include "token.h"

enum class ParserType {
  FLATZINC,
  PROLOG,
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
  bool _open_file;
  bool _open_first_time;
  bool _more_tokens;
  bool _new_line;
  bool _failure;
  
  //! Number of lines read so far
  int _current_line;
  
  //! Delimiter to use to tokenize words
  std::string _delimiters;
  
  //! Other variables needed to move into the file
  std::streampos _curr_pos;
  
  //! Pointers to all tokens parsed so far.
  std::map < size_t, TokenPtr > _map_tokens;
  
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
  
  //! Check whether the parser has failed
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
   */
  virtual TokenPtr get_variable      () = 0;
  virtual TokenPtr get_constraint    () = 0;
  virtual TokenPtr get_search_engine () = 0;
  
  /**
   * Give next Token.
   * A Token is built from a (string) token and represents
   * a semantic object read from the FlatZinc model given
   * in input.
   * It holds other useful info related to the
   * (string) token itself, e.g.,
   * line where the token has been found.
   * If this function is call and no other
   * Token is available it returns nullprt. 
   */
  virtual TokenPtr get_next_content () = 0;
  
  //! Print info
  virtual void print () const = 0;
};


#endif
