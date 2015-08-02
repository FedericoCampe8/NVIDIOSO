//
//  fzn_tokenization.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 04/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "fzn_tokenization.h"
#include "token_var.h"
#include "token_arr.h"
#include "token_con.h"
#include "token_sol.h"

using namespace std;

FZNTokenization::FZNTokenization () {
    _dbg = "FZNTokenization - ";
    set_delimiter ( "\n" );
    add_comment_symb ( "%" );
    unordered_map<string, string> fzn_mapping =
    {
        {"ARR_TOKEN", "array"},
        {"CON_TOKEN", "constraint"},
        {"SOL_TOKEN", "solve"},
        {"VAR_TOKEN",  "var"},
        {"LNS_TOKEN", "lns"}
    };
    set_flatzinc_map ( fzn_mapping );
}//FZNTokenization

FZNTokenization::~FZNTokenization () {
	if ( _parsed_line != nullptr )
	{	
		delete [] _parsed_line;
	}
}//~FZNTokenization


void
FZNTokenization::set_flatzinc_map ( std::unordered_map < std::string, std::string >& m )
{
    _fzn_keywords = m;
}//set_flatzinc_map

void
FZNTokenization::add_flatzinc_word ( std::string key, std::string value )
{
    _fzn_keywords [ key ] = value;
}//add_flatzinc_word

void
FZNTokenization::set_internal_state ( std::string str )
{
	if ( _parsed_line == nullptr )
	{
		_parsed_line = new char[250];
	}
        
	strcpy ( _parsed_line, str.c_str () );
        _c_token = nullptr;
}//set_internal_state

UTokenPtr
FZNTokenization::get_token ()
{
    // Init internal state of the tokenizer
    if ( _c_token == nullptr )
    {
        _c_token = strtok ( _parsed_line , DELIMITERS.c_str() );
    }
    else
    {
        // The line is not completely parsed: get next token.
        _c_token = strtok ( nullptr , DELIMITERS.c_str() );
    }
    
    // Filter _c_token avoiding whitespaces and other useless chars
    clear_line();
  
    /*
     * If the string is terminated: prepare next one.
     * @note: we do not set _more_elements to false.
     * This means that the client will try again
     * to check whether there are other lines to read
     * from the file.
     */
    UTokenPtr current_token ( nullptr );
    if ( _c_token == NULL || skip_line() ) {
        _c_token   = nullptr;
        _need_line = true;
        return move ( current_token );
    }
  
    current_token = analyze_token ();
    if ( current_token == nullptr ) {
        throw  NvdException ( ( _dbg + "Error in tokenization" ).c_str(),
                              __FILE__, __LINE__ );
    }
  
    return move ( current_token );
}//get_token

UTokenPtr
FZNTokenization::analyze_token () 
{

    // Analyze _c_token
    string token_str;
    token_str.assign( _c_token );
    
    /*
     * Check if a keyword has been found.
     * @note valid tokens may contain more than one keyword
     */
    size_t found_arr = token_str.find ( _fzn_keywords[ "ARR_TOKEN" ] );
    size_t found_con = token_str.find ( _fzn_keywords[ "CON_TOKEN" ] );
    size_t found_sol = token_str.find ( _fzn_keywords[ "SOL_TOKEN" ] );
    size_t found_var = token_str.find ( _fzn_keywords[ "VAR_TOKEN" ] );
    size_t found_lns = token_str.find ( _fzn_keywords[ "LNS_TOKEN" ] );
    
    UTokenPtr current_token ( nullptr );
    if ( (found_arr != std::string::npos) && 
    	 (found_con == std::string::npos) )
    {
        current_token = analyze_token_arr ();
    }
    else if ( found_con != std::string::npos )
    {
        current_token = analyze_token_con ();
    }
    else if ( found_sol != std::string::npos )
    {
        current_token = analyze_token_sol ();
    }
    else if ( (found_var != std::string::npos) && 
    		  (found_con == std::string::npos) &&
    		  (found_arr == std::string::npos) )
    {
        current_token = analyze_token_var ();
    }
    else if ( found_lns != std::string::npos )
    {
        current_token =  analyze_token_lns ();
    }
    else
    {
        LogMsg.error ( _dbg + "Not able to parse FlatZinc keyword: " + token_str,
                       __FILE__, __LINE__ );
    }
    return move ( current_token );
}//analyze_token

UTokenPtr
FZNTokenization::analyze_token_arr () 
{

    // Token (pointer) to return
    UTokenPtr t_ptr ( nullptr );
    std::unique_ptr < TokenArr > ptr ( new TokenArr () );
  
    // Skip ARR_TOKEN identifier
    for ( int i = 0; i < _fzn_keywords[ "ARR_TOKEN" ].length(); i++ )
        _c_token++;
    
    // Filter line
    clear_line();
  
    // Use string methods
    std::string token_str ( _c_token );

    /*
     * Read var_type: distinguish between
     * Parameter type and Variable type.
     * par_type: array [index_set] of int, ...
     * var_type: array [index_set] of var int, ...
     */
     bool succeed = ptr->set_token ( token_str );
     if ( !succeed )
     {
     	if ( ptr->is_valid_array () )
     	{// Not valid array declaration
     		LogMsg.error( _dbg + "Parse Error in variable declaration: " + token_str,
     		              __FILE__, __LINE__);
     	}  
     	return move ( t_ptr );
     }
     
     t_ptr = std::move ( ptr );
     return std::move ( t_ptr );
}//analyze_token_arr

UTokenPtr
FZNTokenization::analyze_token_con () 
{
	
  	// Token (pointer) to return
    UTokenPtr t_ptr ( nullptr );
    std::unique_ptr < TokenCon > ptr ( new TokenCon () );
  
  	// Skip CON_TOKEN identifier
  	for ( int i = 0; i < _fzn_keywords[ "CON_TOKEN" ].length(); i++ ) 
  		_c_token++;
  	
  	// Filter line
  	clear_line();
  
  	// Use string methods
  	string token_str;
  	token_str.assign( _c_token );
  	
  	bool succeed = ptr->set_token ( token_str );
     if ( !succeed )
     {
     	LogMsg.error( _dbg + "Parse Error in constraint declaration: " + token_str,
     	              __FILE__, __LINE__);
     	return move ( t_ptr );
     }
     
     t_ptr = std::move ( ptr );
     return std::move ( t_ptr );
}//analyze_token_con

UTokenPtr
FZNTokenization::analyze_token_sol () 
{
	// Token (pointer) to return
    UTokenPtr t_ptr ( nullptr );
    std::unique_ptr < TokenSol > ptr ( new TokenSol () );
  
    // Skip ARR_TOKEN identifier
    for ( int i = 0; i < _fzn_keywords[ "SOL_TOKEN" ].length(); i++ )
        _c_token++;
    
    // Filter line
    clear_line();
  
    // Use string methods
    std::string token_str ( _c_token );

    /*
     * Read var_type: distinguish between
     * Parameter type and Variable type.
     * par_type: array [index_set] of int, ...
     * var_type: array [index_set] of var int, ...
     */
     bool succeed = ptr->set_token ( token_str );
     if ( !succeed )
     {
     	LogMsg.error( _dbg + "Parse Error in solution token: " + token_str,
     	              __FILE__, __LINE__);  
     	return move ( t_ptr );
     }
     
     t_ptr = std::move ( ptr );
     return std::move ( t_ptr );
}//analyze_token_sol

UTokenPtr
FZNTokenization::analyze_token_var () 
{

  	// Token (pointer) to return
    UTokenPtr t_ptr ( nullptr );
    std::unique_ptr < TokenVar > ptr ( new TokenVar () );
  
  	// Skip VAR_TOKEN pattern
  	for ( int i = 0; i < _fzn_keywords[ "VAR_TOKEN" ].length(); i++ ) 
  		_c_token++;
  	
  	// Filter line
  	clear_line();
  
  	// Use string methods
  	std::string token_str ( _c_token );
  	
  	/*
     * Read var_type:
     * var float, var int, ...
     * var int_const..int_const, ...
     * var set of int_const..int_const, ...
     */
     bool succeed = ptr->set_token ( token_str );
     if ( !succeed )
     {
     	LogMsg.error( _dbg + "Parse Error in variable declaration: " + token_str,
     	              __FILE__, __LINE__);
     	return move ( t_ptr );
     }
     
     t_ptr = std::move ( ptr );
     return std::move ( t_ptr );
}//analyze_token_var

UTokenPtr
FZNTokenization::analyze_token_lns () {
  	LogMsg.error( _dbg + "LNS not yet implemented" + _c_token,
    	          __FILE__, __LINE__ );
    	            
  	UTokenPtr t_ptr ( nullptr );
    std::unique_ptr < TokenVar > ptr ( new TokenVar () );
  	return std::move ( t_ptr );
}//analyze_token_lns





