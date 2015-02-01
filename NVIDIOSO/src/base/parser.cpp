//
//  parser.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 01/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "parser.h"

using namespace std;

Parser::Parser ( string ifile ) :
_input_path      ( ifile ),
_open_file       ( false ),
_open_first_time ( true ),
_more_tokens     ( true ),
_new_line        ( true ),
_failure         ( false ),
_current_line    ( -1 ) {
  _dbg         = "Parser - ";
  _delimiters  = " \t\r\n";
}//Parser

Parser::Parser () :
Parser( "" ) {
}//Parser

Parser::~Parser () {
  delete _tokenizer;
}//~Parser

void
Parser::set_input ( string in_file ) {
  if ( _input_path.compare( "" ) == 0 ) {
    _input_path = in_file;
  }
}//set_input

void
Parser::add_delimiter ( string del ) {
  _delimiters += del;
}//add_delimiter

int
Parser::get_current_line () {
  return _current_line;
}//get_line

std::string
Parser::get_next_token () {
  return "";
}//get_next_token

bool
Parser::is_failed () const {
  return _failure;
}//is_failed

bool
Parser::more_tokens () {
  return _more_tokens;
}//has_more_elements

void
Parser::open () {
  if ( !_open_file ) {
    _if_stream = new ifstream( _input_path, ifstream::in );
    if ( !_if_stream->is_open() ) {
      /*
       * Some problems here.
       * @note: we set more elements to false instead of
       * exiting the program.
       * For example, it may happen the case where the file
       * is moved between a series of open() - close() calls.
       * The client wants to preserve the part of the file
       * already read.
       */
      logger->error ( _dbg + "Can't open the file: " + _input_path,
                     __FILE__, __LINE__  );
      _more_tokens = false;
      return;
    }

    if ( _open_first_time ) {
      _curr_pos  = _if_stream->beg;
      _open_first_time  = false;
    }
    _open_file = true;
    
    logger->message( _dbg + "Open file: " + _input_path );
  }
}//open

void
Parser::close () {
  _if_stream->close();
  _open_file = false;
  
  logger->message( _dbg + "Close file: " + _input_path );
}//close

