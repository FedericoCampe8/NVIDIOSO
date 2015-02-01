//
//  tokenization.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 03/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "tokenization.h"

using namespace std;

Tokenization::Tokenization () :
_c_token     ( nullptr ),
_parsed_line ( nullptr ),
_new_line    ( false ),
_need_line   ( true ),
_failed      ( false ) {
  _comment_lines = "";
}//Tokenization

Tokenization::~Tokenization () {
}//~Tokenization

void
Tokenization::add_delimiter ( string add_delim ) {
  DELIMITERS += add_delim;
}//add_delimiter

void
Tokenization::set_delimiter ( string set_delim ) {
  DELIMITERS.assign ( set_delim );
}//set_delimiter

void
Tokenization::add_white_spaces ( string add_ws ) {
  WHITESPACE += add_ws;
}//add_delimiter

void
Tokenization::set_white_spaces ( string set_ws ) {
  WHITESPACE.assign ( set_ws );
}//set_delimiter

bool
Tokenization::need_line () {
  return _need_line;
}//need_line

bool
Tokenization::is_failed () const {
  return _failed;
}//is_failed

void
Tokenization::set_new_tokenizer ( string line ) {
  // Delete previous allocate memory
  if ( _parsed_line != nullptr ) delete [] _parsed_line;
  
  // Allocate memory for the current line
  _parsed_line = new char [ line.length() + 1 ];
  strcpy ( _parsed_line, line.c_str() );
  
  _c_token   = nullptr;
  _need_line = false;
}//set_new_tokenizer

void
Tokenization::add_comment_symb ( string str ) {
  _comment_lines += str;
}//add_comment_symb

void
Tokenization::add_comment_symb ( char chr ) {
  _comment_lines += chr;
}//add_comment_symb

bool
Tokenization::avoid_char ( char c_val ) {
  return true;
}//avoid_char

bool
Tokenization::skip_line () {
  string str_char;
  str_char.assign( _c_token, 1 );
  return ( _comment_lines.find_first_of( str_char.at( 0 ) ) != std::string::npos );
}//avoid_line

bool
Tokenization::skip_line ( std::string line ) {
  return ( _comment_lines.find_first_of( line.at( 0 ) ) != std::string::npos );
}//avoid_line

void
Tokenization::clear_line () {
  if ( _c_token == NULL ) return;
  
  bool found_to_skip = false;
  while ( *_c_token != '\0' ) {
    
    for ( int i = 0; i < WHITESPACE.length(); i++ ) {
      while ( (*_c_token == WHITESPACE.at( i )) && (*_c_token != '\0') ) {
        _c_token++;
        found_to_skip = true;
      }
    }//i
    
    if ( found_to_skip ) {
      found_to_skip = false;
      continue;
    }
    break;
  }
}//clear_line

bool
Tokenization::set_new_line () {
  return true;
}//set_new_line

bool
Tokenization::find_new_line () {
  if ( _new_line ) {
    _new_line = false;
    return true;
  }
  return false;
}//find_new_line


