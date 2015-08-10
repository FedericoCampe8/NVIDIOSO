//
//  fzn_parser.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 03/07/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "fzn_parser.h"
#include "fzn_tokenization.h"
#include "token_arr.h"
#include "token_var.h"

using namespace std;

// @note no delegating constructors gcc < 47
FZNParser::FZNParser () :
    Parser (""),
    _num_tokens ( 0 ) {
    _dbg = "FZNParser - ";

    // Instantiate a Tokenizer for FlatZinc models
    _tokenizer   = new FZNTokenization();
    _tokenizer->add_comment_symb( "%" );
}//FZNParser

FZNParser::FZNParser ( string ifile ) :
Parser ( ifile ),
_num_tokens ( 0 ) {
  _dbg = "FZNParser - ";
  
  // Instantiate a Tokenizer for FlatZinc models
  _tokenizer   = new FZNTokenization();
  _tokenizer->add_comment_symb( "%" );
}//FZNParser

std::string 
FZNParser::replace_bool_vars ( std::string line )
{
	return find_and_replace ( line, "var bool", "var 0..1" );
}//replace_bool_vars

std::string 
FZNParser::replace_bool_vals ( std::string line )
{
	std::string rep = find_and_replace ( line, "true", "1" );
	return find_and_replace ( rep, "false", "0" );
}//replace_bool_vals

std::string 
FZNParser::replace_bool ( std::string line )
{
	std::string rep = replace_bool_vars ( line );
	return replace_bool_vals ( rep );
}//replace_bool

bool
FZNParser::more_aux_arrays () const 
{
	auto it = _lookup_token_table.find ( TokenType::FD_VAR_INFO_ARRAY );
  	if ( it != _lookup_token_table.end() ) 
  	{
    	return ( it->second.size() > 0 );
  	}
  	return false;
}//more_aux_arrays

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
FZNParser::more_constraints () const 
{
	return (more_base_constraints() || more_global_constraints ());
}//more_constraints

bool 
FZNParser::more_base_constraints () const 
{
	auto it = _lookup_token_table.find( TokenType::FD_CONSTRAINT );
  	if ( it != _lookup_token_table.end() ) 
  	{
    	return ( it->second.size() > 0 );
  	}
  	return false;
}

bool 
FZNParser::more_global_constraints () const 
{
	auto it = _lookup_token_table.find( TokenType::FD_GLB_CONSTRAINT );
  	if ( it != _lookup_token_table.end() ) 
  	{
    	return ( it->second.size() > 0 );
  	}
  	return false;
}

bool
FZNParser::more_search_engines () const {
  auto it = _lookup_token_table.find( TokenType::FD_SOLVE );
  if ( it != _lookup_token_table.end() ) {
    return ( it->second.size() > 0 );
  }
  return false;
}//more_search_engines

UTokenPtr 
FZNParser::get_aux_array ()
{
	if ( more_aux_arrays () )
    {
        auto it = _lookup_token_table.find( TokenType::FD_VAR_INFO_ARRAY );
        size_t ptr = it->second[ it->second.size () - 1 ];
        it->second.pop_back();
        _num_tokens--;
        return std::move ( _map_tokens[ ptr ] );
    }//more_constraints
    
    UTokenPtr ptr ( nullptr );
    return std::move ( ptr );
}//get_aux_array

UTokenPtr
FZNParser::get_variable () {
  
    /*
     * Check which variable to return:
     * whether a FD Variable or an Array of variable.
     */
    if ( more_variables () )
    {
        auto it = _lookup_token_table.find( TokenType::FD_VARIABLE );
        if ( it != _lookup_token_table.end() )
        {
            size_t dim = it->second.size ();
            if ( dim > 0 ) {
                size_t ptr_idx = it->second[ dim - 1 ];
                it->second.pop_back();
                _num_tokens--;
                return std::move ( _map_tokens.at( ptr_idx ) );
            }
        }
  
        auto it_aux = _lookup_token_table.find( TokenType::FD_VAR_ARRAY );
        if ( it_aux != _lookup_token_table.end() )
        {
            size_t dim = it_aux->second.size ();
            if ( dim > 0 )
            {
                size_t ptr_idx = it_aux->second[ dim - 1 ];
                it_aux->second.pop_back();
                _num_tokens--;
                return std::move ( _map_tokens.at( ptr_idx ) );
            }
        }
    }//more_variables

    UTokenPtr ptr ( nullptr );
    return std::move ( ptr );
}//get_variable

UTokenPtr
FZNParser::get_constraint () {
    if ( more_base_constraints () )
    {
        auto it = _lookup_token_table.find( TokenType::FD_CONSTRAINT );
        size_t ptr = it->second[ it->second.size () - 1 ];
        it->second.pop_back();
        _num_tokens--;
        return std::move ( _map_tokens[ ptr ] );
    }//more_constraints
    
    if ( more_global_constraints () )
    {
    	auto it = _lookup_token_table.find( TokenType::FD_GLB_CONSTRAINT );
    	size_t ptr = it->second[ it->second.size () - 1 ];
        it->second.pop_back();
        _num_tokens--;
        return std::move ( _map_tokens[ ptr ] );
    }
    
    UTokenPtr ptr ( nullptr );
    return std::move ( ptr );
}//get_constraint

UTokenPtr
FZNParser::get_search_engine () {
    if ( more_search_engines () )
    {
        auto it = _lookup_token_table.find( TokenType::FD_SOLVE );
        size_t ptr = it->second[ it->second.size () - 1 ];
        it->second.pop_back();
        _num_tokens--;
        return std::move ( _map_tokens[ ptr ] );
    }//more_constraints
    
    UTokenPtr ptr ( nullptr );
    return std::move ( ptr );
}//get_search_engine


// Get token (i.e., FlatZinc element)
bool
FZNParser::parse () {
  
    // Open stream (if not already opened)
    if ( !_open_file ) { open(); }
    if ( !_open_file ) { return false; }

    while ( _more_tokens )
    {
        /*
         * Check if previous call of the method started to
         * parse a new line.
         */
        if ( _new_line )
        {
            _current_line++;
            _new_line = false;
        }
  
        /*
         * Check last read:
         * if already read all the string,
         * get a new string to tokenize.
         */
        if ( _tokenizer->need_line() )
        {    
            // Set position on file to the most recent position
            _if_stream->seekg ( _curr_pos );
            string line;
            
            /*
             * Check whether there is another line
             * to parse and, if it is the case, get it.
             */
            if ( getline ( *_if_stream, line ) )
            {
                while ( line.size() == 0 )
                {
                    if ( !getline ( *_if_stream, line ) )
                    {
                        /*
                         * No more line available:
                         * close file, clear everything and exit.
                         */
                        _more_tokens = false;
                        close();
                        return true;
                    }
                }

                // Otherwise continue parsing
                _more_tokens = true;
      
                // Update position on the stream
                _curr_pos = _if_stream->tellg();
      	
				/*
				 * Replace strings in FlatZinc syntax with a 
				 * syntax, structures, more suitable for iNVIDIOSO.
				 */
      			line = replace_bool ( line );
      			
                // Get token
                _tokenizer->set_new_tokenizer( line );
                _new_line  = _tokenizer->find_new_line();
      
                /*
                 * Get token.
                 * @note: if nullptr -> failure in tokenization.
                 */
                UTokenPtr token ( nullptr );
                try
                {
                    token = std::move ( _tokenizer->get_token() );
                }
                catch (...)
                {
                    _failure = true;
                    _more_tokens = false;
                    LogMsg.error ( _dbg + "Not able to get token",
                                   __FILE__, __LINE__ );
                    return false;
                }
      
                // Continue with next token
                if ( token == nullptr ) 
                	continue;
                
                // Store the token
                store_token ( std::move ( token ) );
            }//getline
            else
            {
                /*
                 * No more lines available:
                 * close file, clear everything and exit.
                 */
                _more_tokens = false;
                close();
                return true;
            }
        }//need_line

        // Continue parsing current line
        UTokenPtr token ( nullptr );
        try
        {
            token = std::move ( _tokenizer->get_token() );
        }
        catch (...)
        {       
            _failure = true;
            _more_tokens = false;
            LogMsg.error ( _dbg + "Not able to get token",
                           __FILE__, __LINE__ );
            return false;
        }
        _new_line = _tokenizer->find_new_line();
  
        // Return if nullptr
        if ( token == nullptr )
            continue;
        
        // Store token
        store_token ( std::move ( token ) );
    }//while
    
    return true;
}//get_next_content

void
FZNParser::store_token ( UTokenPtr token ) {
    
    // Check for pointer consistency
    if ( token == nullptr ) return;
  
    // Split token if it is an array of tokens
    if ( token->get_type() == TokenType::FD_VAR_ARRAY )
    {
        TokenArr * arr_ptr = static_cast<TokenArr *> ( token.get() );
        int size = arr_ptr->get_size_arr ();
        
        // Generate "size" tokens corresponding to the array's elements
        for ( int i = 1; i <= size; i++ )
        {
        	// New variable to store (one for each array's element
            unique_ptr < TokenVar > ptr ( new TokenVar () );
      		
      		// Set domain
      		if ( arr_ptr->get_var_dom_type() == VarDomainType::RANGE )
      		{
      			ptr->set_range_domain ( arr_ptr->get_lw_bound_domain (), 
      							  		arr_ptr->get_up_bound_domain () );
      		}
      		else if ( arr_ptr->get_var_dom_type() == VarDomainType::SET )
      		{
      			ptr->set_subset_domain ( arr_ptr->get_subset_domain() );
      			ptr->set_var_dom_type ( arr_ptr->get_var_dom_type () );
      		}
      		else if ( arr_ptr->get_var_dom_type() == VarDomainType::SET_RANGE )
      		{
      			ptr->set_subset_domain ( make_pair( arr_ptr->get_lw_bound_domain(),
                         	   						arr_ptr->get_up_bound_domain() ) );
            	ptr->set_var_dom_type ( arr_ptr->get_var_dom_type () );
      		}
      		else 
      		{
      			LogMsg.error ( _dbg + "Domain not recognized while storing token",
                           	   __FILE__, __LINE__ );
                throw NvdException ( (_dbg + "Domain not recognized while storing token").c_str() );
      		}
      		
      		// Set support var
            if ( arr_ptr->is_support_var () )
            {
                ptr->set_support_var ();
            }
            
            // Set objective var
            if ( arr_ptr->is_objective_var () )
            { 
                ptr->set_objective_var ();
            }
            
            // Set id
            std::string var_id = arr_ptr->get_var_id();
      		var_id += "[";
            ostringstream stream;
            stream << i;
            var_id += stream.str();
            var_id += "]";
            ptr->set_var_id ( var_id );

            // Store the index of the current token in the look_up table
            _lookup_token_table [ arr_ptr->get_type() ].push_back ( _num_tokens );
            
      		// Insert (pointer to) token into map
      		UTokenPtr t_ptr = std::move ( ptr );
      		_map_tokens[ _num_tokens++ ] = std::move ( t_ptr );
      	}
  	}//FD_VAR_ARRAY
  	else 
  	{
    	// Store the index of the current token in the look_up table
    	_lookup_token_table [ token->get_type() ].push_back ( _num_tokens );
    	
    	// Insert (pointer to) token into map
    	_map_tokens[ _num_tokens++ ] = std::move ( token );
  	}
}//store_token

void
FZNParser::print () const {
  	cout << "FZNParser:\n";
  	cout << "Number of tokens: " << _num_tokens << endl;
  	cout << "Variables    : ";
  	if ( more_variables() ) 
  	{
    	auto it     = _lookup_token_table.find( TokenType::FD_VARIABLE );
    	auto it_idx = _lookup_token_table.find( TokenType::FD_VAR_ARRAY );
    	size_t dim = 0;
    	if ( it != _lookup_token_table.end() ) 
    	{
     	 	dim += it->second.size ();
    	}
    	if ( it_idx != _lookup_token_table.end() ) 
    	{
      		dim += it_idx->second.size ();
    	}
    	cout << dim << "\n";
  	}
  	else 
  	{
    	cout << 0 << "\n";
  	}
  	cout << "Constraints  : ";
  	if ( more_constraints () ) 
  	{
    	auto it     = _lookup_token_table.find( TokenType::FD_CONSTRAINT );
    	cout << it->second.size () << "\n";
  	}
  	else 
  	{
    	cout << 0 << "\n";
  	}
  
  	cout << "Search Engine: ";
  	if ( more_search_engines () ) 
  	{
    	auto it     = _lookup_token_table.find( TokenType::FD_SOLVE );
    	cout << it->second.size () << "\n";
  	}
  	else 
  	{
    	cout << 0 << "\n";
 	}
}//print


