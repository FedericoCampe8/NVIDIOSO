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
        {"BOO_TOKEN", "bool"},
        {"CON_TOKEN", "constraint"},
        {"FAL_TOKEN", "false"},
        {"FLO_TOKEN", "float"},
        {"INT_TOKEN", "int"},
        {"LNS_TOKEN", "lns"},
        {"MIN_TOKEN", "minimize"},
        {"MAX_TOKEN", "maximize"},
        {"OAR_TOKEN", "output_array"},
        {"OBJ_TOKEN", "var"},
        {"OUT_TOKEN", "output"},
        {"RAN_TOKEN", ".."},
        {"SAT_TOKEN", "satisfy"},
        {"SET_TOKEN", "set"},
        {"SHO_TOKEN", "show"},
        {"SOL_TOKEN", "solve"},
        {"VAR_TOKEN",  "var"},
        {"VII_TOKEN",  ":: var_is_introduced"}
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
FZNTokenization::analyze_token () {

    // Analyze _c_token
    string token_str;
    token_str.assign( _c_token );
    
    // Check if a keyword has been found
    size_t found_arr = token_str.find ( _fzn_keywords[ "ARR_TOKEN" ] );
    size_t found_con = token_str.find ( _fzn_keywords[ "CON_TOKEN" ] );
    size_t found_sol = token_str.find ( _fzn_keywords[ "SOL_TOKEN" ] );
    size_t found_var = token_str.find ( _fzn_keywords[ "VAR_TOKEN" ] );
    size_t found_lns = token_str.find ( _fzn_keywords[ "LNS_TOKEN" ] );
    
    UTokenPtr current_token ( nullptr );
    if ( found_arr != std::string::npos )
    {
        current_token = analyze_token_arr ();
    }
    else if ( found_con != std::string::npos )
    {
        current_token =  analyze_token_con ();
    }
    else if ( found_sol != std::string::npos )
    {
        current_token = analyze_token_sol ();
    }
    else if ( found_var != std::string::npos )
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
FZNTokenization::analyze_token_arr () {
  
    // Token (pointer) to return
    UTokenPtr t_ptr ( nullptr );
    std::unique_ptr < TokenArr > ptr ( new TokenArr () );
  
    // Skip ARR_TOKEN identifier
    for ( int i = 0; i < _fzn_keywords[ "ARR_TOKEN" ].length(); i++ )
        _c_token++;
    
    // Filter line
    clear_line();
  
    // Use string methods
    string token_str;
    token_str.assign( _c_token );
  
    // Check for support variable
    if ( token_str.find( _fzn_keywords[ "VII_TOKEN" ], 0 ) != std::string::npos )
    {
        ptr->set_support_var ();
    }
  
    // Check for output array
    if ( token_str.find( _fzn_keywords[ "OAR_TOKEN" ], 0 ) != std::string::npos )
    {
        ptr->set_output_arr ();
    }

    /*
     * Read var_type: distinguish between
     * Parameter type and Variable type.
     */
    size_t ptr_idx, ptr_aux;
    ptr_idx = token_str.find ( "var" );
    if ( ptr_idx != std::string::npos )
    {
        // Variable type
        ptr_aux = ptr_idx + 3;
        ptr_idx = token_str.find_first_of( ":", 0 );
        ptr_aux = token_str.find_first_not_of ( " ", ptr_aux );
        while ( (ptr_idx > 0) && (token_str.at(ptr_idx - 1 ) == ' ') ) ptr_idx--;
    
        // Get substring "variable_type" from var "variable_type"
        token_str = token_str.substr ( ptr_aux, ptr_idx - ptr_aux );
    }
    else
    {
        // Parameter type
        ptr_idx = token_str.find ( "of" );
        ptr_aux = ptr_idx + 2;
        ptr_idx     = token_str.find_first_of( ":", 0 );
        ptr_aux = token_str.find_first_not_of ( " ", ptr_aux );
        while ( (ptr_idx > 0) && (token_str.at(ptr_idx - 1 ) == ' ') ) ptr_idx--;

        // Get substring
        token_str = token_str.substr ( ptr_aux, ptr_idx - ptr_aux );
    }
  
    // Find the type
    ptr_aux = token_str.find_first_of ( " " );
    string tkn_type = token_str.substr ( 0, ptr_aux );
  
    /*
     * According to the FlatZinc spec., tkn_type is one of the following:
     * bool, float, int, set, x1..x2.
     */
    if ( tkn_type.compare ( _fzn_keywords[ "BOO_TOKEN" ] ) == 0 )
    {
        ptr->set_boolean_domain ();
    }
    else if ( tkn_type.compare ( _fzn_keywords[ "FLO_TOKEN" ] ) == 0 )
    {
        ptr->set_float_domain ();
    }
    else if ( tkn_type.compare ( _fzn_keywords[ "INT_TOKEN" ] ) == 0 )
    {
        ptr->set_int_domain ();
    }
    else if ( tkn_type.compare ( _fzn_keywords[ "SET_TOKEN" ] ) == 0 )
    {
        /*
         * According to FlatZinc, set could be on of the following:
         * int, x1..x2, {x1, x2, ..., xk}
         */
        ptr->set_subset_domain ( token_str );
    }
    else if ( token_str.find ( _fzn_keywords[ "RAN_TOKEN" ] ) != std::string::npos )
    {
        ptr->set_range_domain ( token_str );
    }
    else if ( token_str.at ( 0 ) == '{' ) {
        ptr->set_subset_domain ( token_str );
    }
    else
    {
        LogMsg.error( _dbg + "Parse Error in variable declaration: " + token_str,
                      __FILE__, __LINE__);
        return move ( t_ptr );
    }

    // Read identifier
    token_str.assign( _c_token );
    ptr_idx = token_str.find_first_of( ":", 0 );
    token_str = token_str.substr ( ptr_idx + 1 );
    string var_id = "";
    for ( int i = 0; i < token_str.size(); i++ )
    {
        char x = token_str[i];
        if ( var_id.size() && x == ' ' ) break;
        if ( x == ' ') continue;
        if ( (x == ';') || (x == ':') )  break;
        var_id += x;
    }
  	
  	// Sanity check
  	assert ( var_id != "" );
  	
    // Set array id
    ptr->set_var_id ( var_id );

    // Read bounds
    ptr_idx = token_str.find_first_of( "[", 0 );
  
    if ( ptr_idx != string::npos )
    {
        ptr_aux = token_str.find_first_of( "]" );
    
        // Set support array and string representing elements
        string support_element = token_str.substr( ptr_idx, ptr_aux - ptr_idx + 1 );
        ptr->set_support_elements ( support_element );
    }
    
    // Set range array
    token_str.assign( _c_token );
    ptr_idx   = token_str.find_first_of( "[", 0 );
    ptr_aux   = token_str.find_first_of( "]", 0 );
    token_str = token_str.substr ( 1, ptr_aux - ptr_idx - 1 );
  
    int lower_bound, upper_bound;
    ptr_aux = token_str.find_first_of( ".", 0 );
    lower_bound = atoi( token_str.substr( 0, ptr_aux ).c_str() );
    ptr_aux = token_str.find_first_of( ":", ptr_aux + 2 );
  
    //Check ' ' before ':'
    token_str = token_str.substr( token_str.find_first_of( ".", 0 ) + 2, ptr_aux );
    ptr_aux = token_str.find_first_of( " ", 0 );
  
    if ( ptr_aux == std::string::npos )
    {
        upper_bound = atoi ( token_str.c_str() );
    }
    else
    {
        upper_bound = atoi ( token_str.substr( 0, ptr_aux ).c_str() );
    }
    ptr->set_array_bounds ( lower_bound, upper_bound );
  
    t_ptr = std::move ( ptr );
    return std::move ( t_ptr );
}//analyze_token_arr

UTokenPtr
FZNTokenization::analyze_token_con () {
	
  	// Token (pointer) to return
    UTokenPtr t_ptr ( nullptr );
    std::unique_ptr < TokenCon > ptr ( new TokenCon () );
  
  	// // Skip CON_TOKEN identifier
  	for ( int i = 0; i < _fzn_keywords[ "CON_TOKEN" ].length(); i++ ) 
  		_c_token++;
  	
  	// Filter line
  	clear_line();
  
  	// Use string methods
  	string token_str;
  	token_str.assign( _c_token );
  
  	// Read constraint identifier
  	size_t ptr_idx, ptr_aux;
  	ptr_idx     = token_str.find_first_of( "(" );
  	ptr_aux = token_str.find_first_of( ")" );
  	if ( (ptr_idx == std::string::npos) ||
       	 (ptr_aux == std::string::npos) ||
       	 (ptr_aux < ptr_idx) ) 
    {
    	LogMsg.error( _dbg + "Constraint not valid" + _c_token,
         	          __FILE__, __LINE__ );
    	return std::move ( t_ptr );
  	}
  
  	ptr->set_con_id( token_str.substr( 0, ptr_idx ) );

  	// Get the expressions that identify the constraint
  	token_str = token_str.substr ( ptr_idx + 1, ptr_aux - ptr_idx - 1 );
  
  	int brk_counter = 0;
  	string expression = "";
  	for ( int i = 0; i < token_str.size(); i++ ) 
  	{
  		char x = token_str[i];	
    	if ( x == '[' ) brk_counter++;
    	if ( x == ']' ) 
    	{
      		expression += x;
      		if ( brk_counter ) 
      		{
        		brk_counter--;
        		if ( brk_counter == 0 ) 
        		{
        			ptr->add_expr ( expression );
		          	expression.assign("");
        		}
      		}
    	}//']'
    	else if ( x == ',' ) 
    	{
      		if ( brk_counter > 0 ) 
      		{
        		expression += x;
      		}
      		else if ( expression.length() ) 
      		{
      			ptr->add_expr ( expression );
	        	expression.assign("");
      		}
    	}
    	else if ( x == ' ' ) 
    	{
      		if ( brk_counter > 0 ) {
        		expression += x;
      		}
      		else 
      		{
        		expression.assign("");
      		}
    	}
    	else 
    	{
      	expression += x;
    	}
  	}//x
  	if ( expression.length() ) 
  	{
  		ptr->add_expr ( expression );
  	}
  
  	t_ptr = std::move ( ptr );
    return std::move ( t_ptr );
}//analyze_token_con

UTokenPtr
FZNTokenization::analyze_token_sol () {

  	// Token (pointer) to return
    UTokenPtr t_ptr ( nullptr );
    std::unique_ptr < TokenSol > ptr ( new TokenSol () );
  	
  	// Skip pattern
  	for ( int i = 0; i < _fzn_keywords[ "SOL_TOKEN" ].length(); i++ ) 
  		_c_token++;
  		
  	// Filter line	
  	clear_line();
  
  	// Use string methods
  	string token_str;
  	token_str.assign( _c_token );
  
  	std::size_t found, found_aux;
  	found = token_str.find( _fzn_keywords[ "SAT_TOKEN" ] );
  	if ( found != std::string::npos ) 
  	{
  		ptr->set_solve_goal ( _fzn_keywords[ "SAT_TOKEN" ] );
  	}
  
  	found = token_str.find( _fzn_keywords[ "MIN_TOKEN" ] );
  	if ( found != std::string::npos ) 
  	{
  		ptr->set_solve_goal ( _fzn_keywords[ "MIN_TOKEN" ] );	
    	string str_aux = token_str.substr ( found + _fzn_keywords[ "MIN_TOKEN" ].size() );
    	string var_to_minimize = "";
    	for ( int i = 0; i < str_aux.size(); i++ ) 
    	{
    		char x = str_aux[i];
      		if ( x == ' ' ) continue;
      		var_to_minimize += x;
    	}
    	
    	// Sanity check 
    	assert ( var_to_minimize != "" );
    	ptr->set_var_goal ( var_to_minimize );
  	}
  
  	found = token_str.find( _fzn_keywords[ "MAX_TOKEN" ] );
  	if ( found != std::string::npos ) 
  	{	
  		ptr->set_solve_goal ( _fzn_keywords[ "MAX_TOKEN" ] );
 	   	string str_aux = token_str.substr ( found + _fzn_keywords[ "MAX_TOKEN" ].size() );
    	string var_to_maximize = "";
    	for ( int i = 0; i < str_aux.size(); i++ ) 
    	{
    		char x = str_aux[i];
      		if ( x == ' ' ) continue;
      		var_to_maximize += x;
    	}
    	
    	// Sanity check 
    	assert ( var_to_maximize != "" );
    	ptr->set_var_goal ( var_to_maximize );
  	}
  
  	// Check annotations
  	found = token_str.find( "::" );
  	if ( found != std::string::npos ) 
  	{
    	found     = token_str.find_first_of ( "(" );
    	found_aux = token_str.find_first_of ( ")" );
    	if ( (found_aux < found) ||
         	 (found     == std::string::npos) ||
         	 (found_aux == std::string::npos) ) 
        {
      		LogMsg.error( _dbg + "Parse Error in solve statement: " + token_str,
                    	  __FILE__, __LINE__);
      		return std::move ( t_ptr );
    	}
    	string strategy = "";
    	for ( int i = 0; i < token_str.size(); i++ ) 
    	{
    		char x = token_str[i];
    		if ( x == '(' ) break;
      		if ( (x == ' ') || (x == ':') ) continue;
      		strategy += x;
    	}	
    	
    	// Sanity check 
    	assert ( strategy != "" );
    	ptr->set_solve_params ( strategy );
    
    	// Other params within "(" and ")"
    	token_str = token_str.substr( found+1, found_aux - found - 1 );
    
    	// Check whether there is a set of variables to label
    	if ( token_str.at ( 0 ) == '[' ) 
    	{
      		token_str = token_str.substr( 1 );
      
      		bool first_insertion = true;
      		int next_position = 0;
      		int brk_counter   = 1;
      		string expression = "";
      		for ( int i = 0; i < token_str.size(); i++ ) 
    		{
    			char x = token_str[i];
        		next_position++;
        		if ( x == '[' ) 
        		{
          			brk_counter++;
          			if ( brk_counter > 1 && first_insertion ) 
          			{
          				ptr->set_solve_params ( expression );
	  	            	first_insertion = false;
          			}
        		}
        		if ( x == ']' ) 
        		{
          			expression += x;
          			if ( brk_counter > 1 ) 
          			{
            			brk_counter--;
            			if ( brk_counter == 1 ) 
            			{
            				ptr->set_solve_params ( expression );
              				expression.assign("");
            			}
          			}
          			else if ( brk_counter == 1 ) 
          			{
            			break;
          			}
        		}//']'
        		else if ( (x == ',') || (x == ' ') ) 
        		{
          			continue;
        		}
        		else 
        		{
          			expression += x;
        		}
      		}//x

      	token_str = token_str.substr( next_position );
      	char * pch;
      	char * c_str = new char[ token_str.length() + 1 ];
      	strncpy ( c_str, token_str.c_str(), token_str.length() );
      	pch = strtok ( c_str, " ," );
      	while ( pch != NULL ) 
      	{
      		ptr->set_solve_params ( pch );
        	pch = strtok ( NULL, " ," );
      	}
      	delete [] c_str;
    	}
    	else 
    	{
      		char * pch;
      		char * c_str = new char[ token_str.length() + 1 ];
      		strncpy ( c_str, token_str.c_str(), token_str.length() );
      		pch = strtok ( c_str, " ," );
      		while ( pch != NULL ) 
      		{
      			ptr->set_solve_params ( pch );
        		pch = strtok ( NULL, " ," );
      		}
      		delete [] c_str;
    	}
  	}
  
  	t_ptr = std::move ( ptr );
    return std::move ( t_ptr );
}//analyze_token_sol

UTokenPtr
FZNTokenization::analyze_token_var () {

  	// Token (pointer) to return
    UTokenPtr t_ptr ( nullptr );
    std::unique_ptr < TokenVar > ptr ( new TokenVar () );
  	return std::move ( t_ptr );
  
  	// Skip VAR_TOKEN pattern
  	for ( int i = 0; i < _fzn_keywords[ "VAR_TOKEN" ].length(); i++ ) 
  		_c_token++;
  	
  	// Filter line
  	clear_line();
  
  	// Use string methods
  	string token_str;
  	token_str.assign( _c_token );
  
  	// Check whether the variable is a support variable
  	if ( token_str.find ( _fzn_keywords[ "VII_TOKEN" ], 0 ) != string::npos ) 
  	{
  		ptr->set_support_var();
  	}
  
  	// Read var_type
  	size_t ptr_idx;
  	ptr_idx = token_str.find_first_of( ":", 0 );
  	while ( (ptr_idx > 0) && (token_str.at(ptr_idx - 1 ) == ' ') ) ptr_idx--;

  	token_str = token_str.substr ( 0, ptr_idx );
  
  	/*
     * According to FlatZinc, tkn_type is one of the following:
     * bool, float, int, set, x1..x2.
   	 */
  	if      ( token_str.compare( _fzn_keywords[ "BOO_TOKEN" ] ) == 0 ) 
  	{	
  		ptr->set_boolean_domain ();
  	}
  	else if ( token_str.compare( _fzn_keywords[ "FLO_TOKEN" ] ) == 0 ) 
  	{
  		ptr->set_float_domain ();
  	}
  	else if ( token_str.compare( _fzn_keywords[ "INT_TOKEN" ] ) == 0 ) 
  	{
  		ptr->set_int_domain ();
  	}
  	else if ( token_str.compare( _fzn_keywords[ "SET_TOKEN" ] ) == 0 ) 
  	{
    	/*
     	 * According to FlatZinc, set could be on of the following:
     	 * int, x1..x2, {x1, x2, ..., xk}
     	 */
     	 ptr->set_subset_domain ( token_str );
  	}
 	else if ( token_str.find ( _fzn_keywords[ "RAN_TOKEN" ] ) != std::string::npos ) 
 	{
 		ptr->set_range_domain ( token_str );
  	}
  	else if ( token_str.at ( 0 ) == '{' ) 
  	{
  		ptr->set_subset_domain ( token_str );
  	}
  	else 
  	{
    	LogMsg.error( _dbg + "Parse Error in variable declaration: " + token_str,
        	          __FILE__, __LINE__);
    	return std::move ( t_ptr );
  	}

  	// Read identifier
  	token_str.assign( _c_token );
  	ptr_idx = token_str.find_first_of( ":", 0 );
  	token_str = token_str.substr ( ptr_idx + 1 );
  	string var_id = "";
  	for ( int i = 0; i < token_str.size(); i++ ) 
  	{
  		char x = token_str[i];
    	if ( var_id.size() && x == ' ' ) break;
    	if ( x == ' ') continue;
    	if ( (x == ';') ||
         	 (x == ':') ) 
        {
      		break;
    	}
    	var_id += x;
  	}
  	
  	// Sanity check
  	assert ( var_id != "" );
  	
  	// Set objective var if found
  	if ( var_id.compare( _fzn_keywords[ "OBJ_TOKEN" ] ) == 0 ) 
  	{
  		ptr->set_objective_var ();
  	}
  	
  	// Set variable id
  	ptr->set_var_id ( var_id );
  
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





