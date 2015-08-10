//
//  tokenization.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/03/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

// Not for gcc < 4.8
//#include <regex>
#include <functional>
#include <algorithm>
#include "tokenization.h"

using namespace std;

Tokenization::Tokenization () :
    _c_token     ( NULL ),
    _parsed_line ( NULL ),
    _new_line    ( false ),
    _need_line   ( true ),
    _failed      ( false ) {
    DELIMITERS = "\t\r\n ";
    WHITESPACE = " \t";
    _comment_lines = "";
}//Tokenization

Tokenization::~Tokenization () {
}//~Tokenization

void
Tokenization::add_delimiter ( string add_delim )
{
    DELIMITERS += add_delim;
}//add_delimiter

void
Tokenization::set_delimiter ( string set_delim )
{
    DELIMITERS.assign ( set_delim );
}//set_delimiter

void
Tokenization::add_white_spaces ( string add_ws )
{
    WHITESPACE += add_ws;
}//add_delimiter

void
Tokenization::set_white_spaces ( string set_ws )
{
    WHITESPACE.assign ( set_ws );
}//set_delimiter

bool
Tokenization::need_line ()
{
    return _need_line;
}//need_line

bool
Tokenization::is_failed () const
{
    return _failed;
}//is_failed

void
Tokenization::set_new_tokenizer ( string line )
{
    // Delete previous allocate memory
    if ( _parsed_line != NULL ) delete [] _parsed_line;
  
    // Allocate memory for the current line
    _parsed_line = new char [ line.length() + 1 ];
    strcpy ( _parsed_line, line.c_str() );
  
    _c_token   = NULL;
    _need_line = false;
}//set_new_tokenizer

void
Tokenization::add_comment_symb ( string str )
{
    _comment_lines += str;
}//add_comment_symb

void
Tokenization::add_comment_symb ( char chr )
{
    _comment_lines += chr;
}//add_comment_symb

bool
Tokenization::avoid_char ( char c_val )
{
    return true;
}//avoid_char

bool
Tokenization::skip_line ()
{
    string str_char ( const_cast< const char * > ( _c_token ) );
    return skip_line ( str_char );
}//skip_line

bool
Tokenization::skip_line ( std::string line ) {
    // Note: the following doesn't work for compilers < 4.8
    //line = std::regex_replace ( line, std::regex("^ +"), "" );
    
    line.erase ( line.begin(),
                 std::find_if ( line.begin(), line.end(),
                                std::bind1st(std::not_equal_to<char>(), ' '))
                 );
                                                                                     
    return ( _comment_lines.find_first_of ( line.at( 0 ) ) != std::string::npos );
}//skip_line

void
Tokenization::clear_line () {
    // Sanity check
    if ( _c_token == NULL ) return;
    
    string str_char ( const_cast< const char * > ( _c_token ) );

    // Checks if the line starts with a "comment" symbol
    string white_sp = "^[";
    for ( int i = 0; i < WHITESPACE.length()-1; i++ )
    {
        white_sp += WHITESPACE [ i ];
        white_sp += "|";
    }
    white_sp += WHITESPACE [ WHITESPACE.length()-1 ];
    white_sp += "]+";

    // Following not working for gcc < 4.8
    //str_char = std::regex_replace ( str_char, std::regex( white_sp ), "" );
    for ( int i = 0; i < WHITESPACE.length(); i++ )
        str_char.erase ( str_char.begin(),
                         std::find_if ( str_char.begin(), str_char.end(),
                                        std::bind1st(std::not_equal_to<char>(), WHITESPACE[i] ))
                     );
    char * ptr = _c_token;
    for ( int i = 0; i < str_char.size(); i++ )
    {
        *ptr++ = str_char[i];
    }
    *ptr = '\0';
}//clear_line

bool
Tokenization::set_new_line ()
{
    return true;
}//set_new_line

bool
Tokenization::find_new_line ()
{
    if ( _new_line )
    {
        _new_line = false;
        return true;
    }
    return false;
}//find_new_line


