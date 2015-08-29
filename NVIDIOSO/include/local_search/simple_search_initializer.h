//
//  simple_search_initializer.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/25/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a simple search initializer.
//  A search initializer is responsable for the initialization of the (local) search process.
//

#ifndef __NVIDIOSO__simple_search_initializer__
#define __NVIDIOSO__simple_search_initializer__

#include "search_initializer.h"

class SimpleSearchInitializer : public SearchInitializer {
protected:

	//! Debug string info
	std::string _dbg;
	
	/**
	 * Hash table of variables ids and Boolean value stating 
	 * whether a variable is initialized or not.
	 * @note This hash table is used to speedup lookup methods.
	 */
	std::unordered_map < int, bool > _initialized_variables;
	
	/**
	 * Hash table of variables ids and integer values 
	 * used to store the initialization performed by the initialize method.
	 * @note This hash table is used to speedup lookup methods.
	 */
	std::unordered_map < int, int > _initialized_values;
	
	//! Array of (pointers to) variables to initialize
  	std::vector< Variable* > _fd_variables;
  	
public:

	/**
	 * Constructor for a simple search initializer.
	 * @param vars array of (pointers to) variables to initialize.
	 */
	SimpleSearchInitializer ( std::vector< Variable* > vars );
	
	~SimpleSearchInitializer ();
	
	/**
   	 * Set the list of variables for which an initialization is required.
   	 * @param vars a vector of references to variables.
   	 * @note if an array has already been set, this will substitute the current set 
   	 *       of variables internally handled by this search initializer.
   	 *       A new call to the initialize function will be required.
   	 */
  	void set_variables ( std::vector < Variable* > vars ) override;
	
	/**
	 * Returns true if the variable is considered for initialization
	 * as internal variable in the state of this initializer.
	 * @param var pointer to the variable to query
	 * @return true if var is considered for initialization by this initializer,
	 *         false otherwise.
	 */
	 bool is_being_initializer ( Variable * var ) const override;
	 
	/**
   	 * Returns true if the variable is initialized.
   	 * @param var pointer to the variable to check initialization
   	 * @note if var is not in the set of initialized variables, returns false
   	 */
  	bool is_initialized ( Variable * var ) const override;
	
	//! Print initialization of variables
	virtual void print_initialization () const override;
};

#endif /* defined(__NVIDIOSO__simple_search_initializer__) */
