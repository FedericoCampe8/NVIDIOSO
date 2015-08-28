//
//  token_cstore.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/27/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  Token to represent info for constraint store.
//  @note This does not belong to the FlatZinc specification.
//

#ifndef NVIDIOSO_token_cstore_h
#define NVIDIOSO_token_cstore_h

#include "token.h"

class TokenCStore : public Token {
protected:
	//! Specifies whether the constraint store should be specialized for local search
	bool _on_local_search;
	
public:
  TokenCStore ();
  
  bool set_token ( std::string& token_string ) override;
  
  void set_on_local_search ();
  
  bool on_local_search () const;
  
  //! Print info methods
  virtual void print () const override;
};


#endif
