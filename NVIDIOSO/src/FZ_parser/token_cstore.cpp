//
//  token_cstore.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/27/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "token_cstore.h"

using namespace std;

TokenCStore::TokenCStore () :
Token ( TokenType::FD_CONSTRAINT_STORE ) {
  _dbg = "TokenCStore - ";
  _on_local_search = false;
}//TokenCStore

bool 
TokenCStore::set_token ( std::string& token_str )
{
	if ( token_str == "local_search" )
	{
		_on_local_search = true;
	}
	return true;
}//set_token

void 
TokenCStore::set_on_local_search ()
{
	_on_local_search = true; 
}//set_on_local_search

bool 
TokenCStore::on_local_search () const
{
	return _on_local_search;
}//on_local_search

void
TokenCStore::print () const 
{
	cout << "TokenCStore:\n";
	if ( _on_local_search )
	{
		cout << "On local search\n";
	}
	else
	{
		cout << "On complete search\n";
	}
}//print



