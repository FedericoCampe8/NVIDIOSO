//
//  neighborhood_heuristic.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/23/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "neighborhood_heuristic.h"
#include "metric_inc.h"

using namespace std;

NeighborhoodHeuristic::NeighborhoodHeuristic ( std::vector< Variable* >& vars,
 											   VariableChoiceMetric * var_cm,
                                   			   ValueChoiceMetric    * val_cm ) :
    SimpleHeuristic ( vars, var_cm, val_cm ) {
	_dbg = "NeighborhoodHeuristic -";
  	if ( _value_metric == nullptr )
  	{
  		_value_metric = new InDomainMin ();
  	}
}

NeighborhoodHeuristic::~NeighborhoodHeuristic () {
	delete _variable_metric;
	delete _value_metric;
	_variable_metric = nullptr;
	_value_metric    = nullptr;
}//~ NeighborhoodHeuristic

int  
NeighborhoodHeuristic::get_next_index () const
{
	int idx = _current_index + 1;
	if ( idx >= _fd_variables.size() )
	{
		return -1;
	}
	return idx;
}//get_next_index

int
NeighborhoodHeuristic::set_index ( int current_index )
{
	// Sanity check 
  	if ( (current_index < 0) || (current_index >= _fd_variables.size()) ) 
  	{
		std::ostringstream s;
    	s << current_index << " / " <<  _fd_variables.size();
    	throw  NvdException ( (_dbg + " No consistent current_index." + s.str()).c_str() );
  	}
  	
  	int prev_index = _current_index;
	_current_index = current_index;
	
	return prev_index;
}//get_current_index
 
void 
NeighborhoodHeuristic::set_search_variables ( std::vector< Variable* >& vars )
{
	_current_index = 0;
 	_fd_variables = vars;
}//set_search_variables
 
 void 
 NeighborhoodHeuristic::print () const
 {
 	std::cout << "NeighborhoodHeuristic\n";
  	if ( _value_metric != nullptr ) 
  	{
    	std::cout << "\tValue choice metric:   \t";
    	_value_metric->print();
  	}
  	if ( _variable_metric != nullptr ) 
  	{
    	std::cout << "\tVariable choice metric:\t";
    	_variable_metric->print();
 	}
 }//print