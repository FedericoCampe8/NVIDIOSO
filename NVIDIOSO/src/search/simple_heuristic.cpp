//
//  simple_heuristic.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/08/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "simple_heuristic.h"

SimpleHeuristic::SimpleHeuristic ( std::vector< Variable* > vars,
                                   VariableChoiceMetric * var_cm,
                                   ValueChoiceMetric    * val_cm ) {
  _dbg = "SimpleHeuristic -";
  _fd_variables    = vars;
  _variable_metric = var_cm;
  _value_metric    = val_cm;
}//SimpleHeuristic

SimpleHeuristic::~SimpleHeuristic () {
  _fd_variables.clear();
  delete _variable_metric;
  delete _value_metric;
}//~SimpleHeuristic

void 
SimpleHeuristic::placeSearchVariable ( int searchPosition, int variablePosition )
{
	std::iter_swap ( _fd_variables.begin() + searchPosition,
                     _fd_variables.begin() + variablePosition );	
}//placeSearchVariable

Variable *
SimpleHeuristic::get_choice_variable ( int index ) 
{
  // Sanity check
  if ( (index < 0) || (index >= _fd_variables.size()) ) {
    return nullptr;
  }
  
  size_t last_idx = _fd_variables.size();
  Variable * selected_var;
  
  // Find the first "available" variable to label
  do 
  {
	selected_var = _fd_variables[ index ];
  } while ( (selected_var->is_singleton()) && (++index < last_idx) );
	
  // Consistency check: no more vars to label
  if ( index == last_idx ) return nullptr;
  
  /*
   * Sanity check:
   * if there is only one variable left or no variable metric
   * has been provided, then return current selected variable.
   */
  if ( (index + 1 == last_idx) || (_variable_metric == nullptr) ) 
  {// Update current idx
	_current_index = index;
    return selected_var;
  }
  
  /*
   * Check for a "better" variable among the remaining variables.
   * Better variable means a variable with a better metric value
   * calculated with the metric given in input to this heuristic.
   * @note here we use metric_position to save the position of the
   *       variable to return which has a better metric than the
   *       current selected one (with the above do-while loop).
   */
  Variable * other_var;
  int    comparison      = 0;
  int    metric_position = index;
  double metric_value    = _variable_metric->metric_value ( selected_var );
  
  for ( int i = index + 1; i < last_idx; i++ ) 
  {
	other_var = _fd_variables [ i ];
 
    /*
     * The following check skips singleton variables (no point to label them)
     * and updates the position metric_position accordingly.
     */
    if ( other_var->is_singleton () ) 
    {  
      // Next var is singleton, go ahead
      if ( index == metric_position ) 
      {
      	placeSearchVariable ( index, i );
    	metric_position = i;
        index++;
      }
      else 
      {
        // Find the first not singleton variable
        while ( index < i && _fd_variables[index]->is_singleton() ) index++;
        
        // Not singleton, check if index is the current position
        if ( index != i ) 
        {
          /*
           * Index differs from the current position, but
           * the corresponding var is not singleton.
           * Update the position of the metric.
           */
          if ( index == metric_position ) 
          {
          	placeSearchVariable ( index, i );
          	metric_position = i;
          }
          else
          {
          	placeSearchVariable ( index, i );
          }
          
          // Update the index in order to consider next variable
          index++;
        }
      }
      
      continue;
    }
    
    /*
     * Other_var is not singleton,
     * compare the selected variable (metric_value) with the other 
     * not singleton variable and select the one with the
     * best metric value (see: "variable_choice_metric.h"):
     * -1 - metric_value is smaller
     *  0 - equal
     *  1 - metric_value is larger 
     */
    comparison = _variable_metric->compare ( metric_value, other_var );
    
    if ( comparison < 0 ) 
    {// metric_value is smaller, update it with the new larger value (i.e., better choice)
      
      metric_position = i;
      metric_value    = _variable_metric->metric_value ( other_var );
    }
    
    /*
     * If selected_var >= other_var continue in order to find
     * the next larger value if there are any.
     */
  }//i
  
  // Swap the values if index is different from metric
  if ( index != metric_position ) 
  {
  	placeSearchVariable ( index, metric_position );
  }
  
  // Update current index
  _current_index = index;
  return _fd_variables[ index ];
}//get_choice_variable


int
SimpleHeuristic::get_choice_value () 
{  
  // Consistency check
  if ( (_current_index < 0) ||
       (_current_index >= _fd_variables.size()) ) {
    std::ostringstream s;
    s << _current_index << " / " <<  _fd_variables.size();
    throw  NvdException ( (_dbg + " No consistent index." + s.str()).c_str() );
  }
  
  return _value_metric->metric_value ( _fd_variables[ _current_index ] );
}//get_choice_value

void
SimpleHeuristic::print () const {
  std::cout << "Heuristic\n";
  if ( _value_metric != nullptr ) {
    std::cout << "\tValue choice metric:   \t";
    _value_metric->print();
  }
  if ( _variable_metric != nullptr ) {
    std::cout << "\tVariable choice metric:\t";
    _variable_metric->print();
  }
}//print
