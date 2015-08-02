//
//  global_constraint_register.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 30/07/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a register for global constraints.
//  It stores (pointers to) functions that create (global) constraints
//  according to the given input string name of the constraint.
//

/**
 * File with names of global constraints.
 * unordered_map with key = name (string)
 * and value = pointer to global constraint.
 * Init: read the file and fill the matrix with names and instances.
 * Whenever a constraint name is parsed in input, check the map.
 * When match, return a clone of the global constraint and set its vars/args.
 * Use a "register" class: a private unordered map <string, GlobalConstraint> _register;
 * With a method "register" or a constructor:
 * void register () 
 * {
 * 		_register[ "abs_value" ] = new ABSValue ();
 *      ....
 * }
 * The register can be a singleton.
 * Then, parsing a constraint: Constraint * c = register.get ( "alldifferent" );
 * returns a COPY/CLONE of the "alldifferent" global constraint.
 * Then, c.set ( vars, args );
 */

#ifndef __NVIDIOSO__global_constraint_register__
#define __NVIDIOSO__global_constraint_register__

#include "global_constraint.h"

typedef GlobalConstraint* (*poster) ( std::string constraint_name );

class GlobalConstraintRegister {
private:
  
	std::unordered_map<std::string, poster> _register;
  
  	/**
  	 * Add the poster (function to create the global constraint) p
  	 * to the map, with key name, i.e., the name of the global constraint.
  	 * @param name name of the global constraint.
  	 * @param p poster, function to instantiate a new global constraint.
  	 */
  	void add ( std::string name, poster p );
  	
  	//! Init function, fill the register (hash table) with posters
  	void fill_register ();
  	
  	// Singleton instance: private constructor
  	GlobalConstraintRegister ();
  	
public:
	GlobalConstraintRegister ( const GlobalConstraintRegister& other )            = delete;
  	GlobalConstraintRegister& operator= ( const GlobalConstraintRegister& other ) = delete;
  	
  	virtual ~GlobalConstraintRegister();
  	
  	//! Constructor get (static) instance
  	static GlobalConstraintRegister& get_instance () 
  	{
    	static GlobalConstraintRegister register_instance;
    	return register_instance;
  	}//get_instance
  	
  	//! Given the string name of the global constraint, return an instance of it
  	GlobalConstraintPtr get_global_constraint ( std::string glb_constraint_name);
  
};

#endif /* defined(__NVIDIOSO__global_constraint_register__) */
