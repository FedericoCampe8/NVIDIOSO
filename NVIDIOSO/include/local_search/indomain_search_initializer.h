//
//  indomain_search_initializer.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/25/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a simple search initializer where variables are
//  initialized with random values belonging to their domains.
//

#ifndef __NVIDIOSO__indomain_search_initializer__
#define __NVIDIOSO__indomain_search_initializer__

#include "simple_search_initializer.h"

enum class InDomainInitType {
  INDOMAIN_MIN,
  INDOMAIN_MAX,
  INDOMAIN_RAN,
  OTHER
};

class InDomainSearchInitializer : public SimpleSearchInitializer {  	
protected:
	InDomainInitType _initialization_type;
	
public:

	/**
	 * Constructor for a simple search initializer.
	 * @param vars array of (pointers to) variables to initialize.
	 * @param indomain type of initialization, default is random initialzation (i.e., random values
	 *        from the domains of the variables currently set).
	 */
	InDomainSearchInitializer ( std::vector< Variable* > vars, InDomainInitType indom_t = InDomainInitType::INDOMAIN_RAN );
	
	~InDomainSearchInitializer ();
	
	//! Set initialization type to use to initialize variable 
	void set_initialization_type ( InDomainInitType indom_t );
	
	//! Get current type of initialization used
	InDomainInitType get_initialization_type () const;
	
	//! Initialize the variables with random values from their domains.
	void initialize () override;
	
	void print () const override;
};

#endif /* defined(__NVIDIOSO__indomain_search_initializer__) */
