//
//  fzn_parser.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 03/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "fzn_parser.h"
#include "fzn_tokenization.h"
#include "token_arr.h"
#include "token_var.h"

using namespace std;

FZNParser::FZNParser () :
FZNParser ( "" ) {
}//FZNParser

FZNParser::FZNParser ( string ifile ) :
Parser ( ifile ),
_num_tokens ( 0 ) {
  _dbg = "FZNParser - ";
  // Instantiate a Tokenizer for FlatZinc models
  _tokenizer   = new FZNTokenization();
  _tokenizer->add_comment_symb( "%" );
}//FZNParser

bool
FZNParser::more_variables () const {
  auto it     = _lookup_token_table.find ( TokenType::FD_VARIABLE );
  auto it_aux = _lookup_token_table.find ( TokenType::FD_VAR_ARRAY );
  
  size_t dim = 0;
  if ( it     != _lookup_token_table.end() ) dim += it->second.size();
  if ( it_aux != _lookup_token_table.end() ) dim += it_aux->second.size();
  
  return ( dim > 0 );
}//more_variables

bool
FZNParser::more_constraints () const {
  auto it = _lookup_token_table.find( TokenType::FD_CONSTRAINT );
  if ( it != _lookup_token_table.end() ) {
    return ( it->second.size() > 0 );
  }
  return false;
}//more_constraints

bool
FZNParser::more_search_engines () const {
  auto it = _lookup_token_table.find( TokenType::FD_SOLVE );
  if ( it != _lookup_token_table.end() ) {
    return ( it->second.size() > 0 );
  }
  return false;
}//more_search_engines

TokenPtr
FZNParser::get_variable () {
  
  /*
   * Check which variable to return:
   * whether a FD Variable or an Array of variable.
   */
  if ( more_variables () ) {
    auto it = _lookup_token_table.find( TokenType::FD_VARIABLE );
    if ( it != _lookup_token_table.end() ) {
      size_t dim = it->second.size ();
      if ( dim > 0 ) {
        size_t ptr_idx = it->second[ dim - 1 ];
        it->second.pop_back();
        _num_tokens--;
        return _map_tokens.at( ptr_idx );
      }
    }
  
    auto it_aux = _lookup_token_table.find( TokenType::FD_VAR_ARRAY );
    if ( it_aux != _lookup_token_table.end() ) {
      size_t dim = it_aux->second.size ();
      if ( dim > 0 ) {
        size_t ptr_idx = it_aux->second[ dim - 1 ];
        it_aux->second.pop_back();
        _num_tokens--;
        return _map_tokens.at( ptr_idx );
      }
    }
  }//more_variables
    
  return nullptr;
}//get_variable

TokenPtr
FZNParser::get_constraint () {
  if ( more_constraints () ) {
    auto it = _lookup_token_table.find( TokenType::FD_CONSTRAINT );
    size_t ptr = it->second[ it->second.size () - 1 ];
    it->second.pop_back();
    _num_tokens--;
    return _map_tokens[ ptr ];
  }//more_constraints
  return nullptr;
}//get_constraint

TokenPtr
FZNParser::get_search_engine () {
  if ( more_search_engines () ) {
    auto it = _lookup_token_table.find( TokenType::FD_SOLVE );
    size_t ptr = it->second[ it->second.size () - 1 ];
    it->second.pop_back();
    _num_tokens--;
    return _map_tokens[ ptr ];
  }//more_constraints
  return nullptr;
}//get_search_engine


// Get token (i.e., FlatZinc element)
TokenPtr
FZNParser::get_next_content () {
  
  // Open stream (if not already opened)
  if ( !_open_file ) { open(); }
  if ( !_open_file ) { return nullptr; }
  
  /*
   * Check if previous call of the method started to
   * parse a new line.
   */
  if ( _new_line ) {
    _current_line++;
    _new_line = false;
  }
  
  /*
   * Check last read:
   * if already read all the string,
   * get a new string to tokenize.
   */
  if ( _tokenizer->need_line() ) {
    /// Set position on file to the most recent position
    _if_stream->seekg ( _curr_pos );
    string line;
    /*
     * Check whether there is another line
     * to parse and, if it is the case, get it.
     */
    if ( getline ( *_if_stream, line ) ) {
      while ( line.size() == 0 ) {
        if ( !getline ( *_if_stream, line ) ) {
          /*
           * No more line available:
           * close file, clear everything and exit.
           */
          _more_tokens = false;
          close();
          return nullptr;
        }
      }
      _more_tokens = true;
      
      // Update position
      _curr_pos = _if_stream->tellg();
      
      // Get token
      _tokenizer->set_new_tokenizer( line );
      _new_line  = _tokenizer->find_new_line();
      
      /*
       * Get token.
       * @note: if nullptr -> failure in tokenization.
       */
      TokenPtr token;
      try {
        token = _tokenizer->get_token();
      } catch (...) {
        _failure = true;
        _more_tokens = false;
        return nullptr;
      }
      
      // Return if nullptr
      if ( token == nullptr ) { return nullptr; }
      // Store the token
      store_token ( token );
      // Return the token
      return token;
    }//getline
    else {
      
      /*
       * No more line available:
       * close file, clear everything and exit.
       */
      _more_tokens = false;
      close();
      return nullptr;
    }
  }//need_line
  
  TokenPtr token;
  try {
    token = _tokenizer->get_token();
  } catch (...) {
    _failure = true;
    _more_tokens = false;
    return nullptr;
  }
  _new_line = _tokenizer->find_new_line();
  
  // Return if nullptr
  if ( token == nullptr ) { return nullptr; }
  // Store token
  store_token ( token );
  // Everything went well: return the content.
  return token;
}//get_next_content

void
FZNParser::store_token ( TokenPtr token ) {
  // Check for pointer consistency
  if ( token == nullptr ) return;
  
  // Split token if it is an array of tokens
  if ( token->get_type() == TokenType::FD_VAR_ARRAY ) {
    int size = (std::static_pointer_cast<TokenArr>( token ))->get_size_arr ();
    
    // Generate size variable tokens
    for ( int i = 1; i <= size; i++ ) {
      
      // Token (pointer) to return
      TokenPtr tkn_ptr = make_shared<TokenVar> ();
      
      // Set domain
      if ( (std::static_pointer_cast<TokenArr>( token ))->get_var_dom_type() == VarDomainType::RANGE ) {
        (std::static_pointer_cast<TokenVar>( tkn_ptr ))->
        set_range_domain (
                          (std::static_pointer_cast<TokenArr>( token ))->get_lw_bound_domain(),
                          (std::static_pointer_cast<TokenArr>( token ))->get_up_bound_domain()
                          );
      }
      else if ( (std::static_pointer_cast<TokenArr>( token ))->get_var_dom_type() == VarDomainType::SET ) {
        
        (std::static_pointer_cast<TokenVar>( tkn_ptr ))->
        set_subset_domain ( (std::static_pointer_cast<TokenArr>( token ))->get_subset_domain() );
      }
      else if ( (std::static_pointer_cast<TokenArr>( token ))->get_var_dom_type() == VarDomainType::SET_RANGE ) {
        (std::static_pointer_cast<TokenVar>( tkn_ptr ))->
        set_subset_domain (make_pair(
                           (std::static_pointer_cast<TokenArr>( token ))->get_lw_bound_domain(),
                           (std::static_pointer_cast<TokenArr>( token ))->get_up_bound_domain()
                                     ));
      }
      
      (std::static_pointer_cast<TokenVar>( tkn_ptr ))->set_var_dom_type ( (std::static_pointer_cast<TokenArr>( token ))->get_var_dom_type() );
      
      // Set objective var
      if ( (std::static_pointer_cast<TokenArr>( token ))->is_support_var () ) {
        (std::static_pointer_cast<TokenVar>( tkn_ptr ))->set_support_var ();
      }
      
      // Set support var
      if ( (std::static_pointer_cast<TokenArr>( token ))->is_objective_var () ) {
        (std::static_pointer_cast<TokenVar>( tkn_ptr ))->set_objective_var ();
      }
      
      // Set variable id
      string var_id = (std::static_pointer_cast<TokenArr>( token ))->get_var_id();
      var_id += "[";
      ostringstream stream;
      stream << i;
      var_id += stream.str();
      var_id += "]";
      (std::static_pointer_cast<TokenVar>( tkn_ptr ))->set_var_id ( var_id );
      
      // Store the index of the current token in the look_up table
      _lookup_token_table [ tkn_ptr->get_type() ].push_back ( _num_tokens );
      // Insert (pointer to) token into map
      _map_tokens[ _num_tokens++ ] = tkn_ptr;
    }
  }//FD_VAR_ARRAY
  else {
    // Store the index of the current token in the look_up table
    _lookup_token_table [ token->get_type() ].push_back ( _num_tokens );
    // Insert (pointer to) token into map
    _map_tokens[ _num_tokens++ ] = token;
  }
}//store_token

void
FZNParser::print () const {
  
  cout << "FZNParser:\n";
  cout << "Number of tokens: " << _num_tokens << endl;
  cout << "Variables    : ";
  
  if ( more_variables() ) {
    auto it     = _lookup_token_table.find( TokenType::FD_VARIABLE );
    auto it_idx = _lookup_token_table.find( TokenType::FD_VAR_ARRAY );
    size_t dim = 0;
    if ( it != _lookup_token_table.end() ) {
      dim += it->second.size ();
    }
    if ( it_idx != _lookup_token_table.end() ) {
      dim += it->second.size ();
    }
    cout << dim << "\n";
  }
  else {
    cout << 0 << "\n";
  }
  cout << "Constraints  : ";
  if ( more_constraints () ) {
    auto it     = _lookup_token_table.find( TokenType::FD_CONSTRAINT );
    cout << it->second.size () << "\n";
  }
  else {
    cout << 0 << "\n";
  }
  
  cout << "Search Engine: ";
  if ( more_search_engines () ) {
    auto it     = _lookup_token_table.find( TokenType::FD_SOLVE );
    cout << it->second.size () << "\n";
  }
  else {
    cout << 0 << "\n";
  }
}//print


