//
//  token.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 03/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Token element representing elements of a model:
//  this object represent variables, domains and constraint
//  plus related info.
//  It is a "general" token, a base class for all tokens.
//

#ifndef NVIDIOSO_token_h
#define NVIDIOSO_token_h

#include "globals.h" 

class Token;
typedef std::unique_ptr<Token> UTokenPtr;
typedef std::shared_ptr<Token> TokenPtr;

enum class TokenType {
  FD_VARIABLE,
  FD_VAR_ARRAY,
  FD_DOMAIN,
  FD_CONSTRAINT,
  FD_SOLVE,
  FD_LNS,
  OTHER
};
 
class Token {
private:
    //! Token's id
    int _id;
  
protected:
    std::string _dbg;

    //! Specifies the type of token (e.g., FD_VARIABLE) 
    TokenType _tkn_type;
  
public:
    Token ();
    Token ( TokenType );

    virtual ~Token();
    
    // Get/Set methods
    int get_id () const;
  
    void      set_type ( TokenType );
    TokenType get_type () const;
  
    //! Print info about the current token
    virtual void print () const;
};

#endif
