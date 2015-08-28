//
//  search_initializer.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/22/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class declares the interface for a search initializer.
//  A search initializer is responsable for the initialization of the (local) search process.
//  @note Generally there are many types of initializations, for example:
//        1 - From a solution found by complete
//        2 - From file
//        3 - From random assignment
//        ... 
//

#ifndef __NVIDIOSO__search_initializer__
#define __NVIDIOSO__search_initializer__

#include "globals.h"
#include "variable.h"

class SearchInitializer;
typedef std::unique_ptr<SearchInitializer> SearchInitializerUPtr;
typedef std::shared_ptr<SearchInitializer> SearchInitializerSPtr;

class SearchInitializer {
public:

	virtual ~SearchInitializer () {};
	
	/**
	 * Returns the initialization value selected for the variable var.
	 * @param var pointer to the initialized variable.
	 * @note var should be in the list of variables currently initialized 
	 *       by this SearchInitializer.
	 */
	template<typename T> T 
	initialization_value ( Variable * var ) const
	{
		// Sanity check
		assert ( var != nullptr );
		
		if ( !is_initialized ( var ) )
		{
			std::string err_msg { "SearchInitializer::initialization_value variable not initialized" }; 
			throw NvdException ( err_msg.c_str() );
		}
		
		return var->domain_iterator->min_val ();
	}//initialization_value
	
	/**
   	 * Set the list of variables for which an initialization is required.
   	 * @param vars a vector of references to variables.
   	 */
  	virtual void set_variables ( std::vector < Variable* > vars ) = 0;
	
	//! Initialize the variables set internally to this initializer.
	virtual void initialize () = 0;
	
	/**
	 * Returns true if the variable is considered for initialization
	 * as internal variable in the state of this initializer.
	 * @param var pointer to the variable to query
	 * @return true if var is considered for initialization by this initializer,
	 *         false otherwise.
	 */
	 virtual bool is_being_initializer ( Variable * var ) const = 0;
	 
	/**
   	 * Returns true if the variable is initialized.
   	 * @param var pointer to the variable to check initialization
   	 * @return true is var is initialized, false otherwise
   	 * @note if var is not in the set of initialized variables, returns false
   	 */
  	virtual bool is_initialized ( Variable * var ) const = 0;
	
	//! Print initialization of variables
	virtual void print_initialization () const = 0;
	
	//! Print info
	virtual void print () const = 0;
};

#endif /* defined(__NVIDIOSO__search_initializer__) */
