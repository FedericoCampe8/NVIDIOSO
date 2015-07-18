//
//  token.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 03/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "token.h"

Token::Token () {
    // Default token OTHER 
    _tkn_type = TokenType::OTHER;
    _id  = glb_id_gen->new_int_id ();
    _dbg = "Token - ";
}//Token

Token::Token ( TokenType tp ) {
    _tkn_type = tp;
    _id = glb_id_gen->new_int_id ();
    _dbg = "Token - ";
}//Token

Token::~Token () {
}//~Token

int
Token::get_id () const {
  return _id;
}//get_id

void
Token::set_type ( TokenType tp ) {
    if ( _tkn_type != TokenType::OTHER )
    {
        _tkn_type = tp;
    }
}//set_type

TokenType
Token::get_type () const {
    return _tkn_type;
}//get_type

void
Token::print () const {
    std::cout << "Default token object\n";
}//print
