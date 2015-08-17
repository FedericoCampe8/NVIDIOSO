//
//  cuda_concrete_list.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/15/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "cuda_concrete_list.h"

CudaConcreteDomainList::CudaConcreteDomainList ( size_t size, int min, int max ) :
	CudaConcreteDomain ( size ),
	_num_pairs         ( 0 ),
	_domain_size       ( 0 ) {
  	_dbg = "CudaConcreteDomainList - ";
  
	// Empty domain if not consistent
  	if ( max < min ) 
  	{
    	set_empty ();
    	return;
  	}

  	_max_allowed_pairs = _num_chunks / 2;
  	if ( _max_allowed_pairs < 1 ) 
  	{
    	set_empty ();
    	return;
  	}

	// Initialize bitmap
  	for ( int i = 0; i < _num_chunks; i++ ) 
  	{
    	_concrete_domain[ i ] = 0x0;
  	}
  
  	// Set current lower/upper bounds
  	_lower_bound = min;
  	_upper_bound = max;
  
  	// Set initial lower/upper bounds
  	_init_lower_bound = _lower_bound;
  	_init_upper_bound = _upper_bound;
  
  	// Add the new pair of bounds to the list
  	add ( min, max );
}//CudaConcreteDomainList
 
void
CudaConcreteDomainList::set_domain ( void * const domain, int rep, int min, int max, int dsz )
{
  CudaConcreteDomain::set_domain( domain, rep, min, max, dsz );
  _domain_size = dsz;
  _num_pairs   = rep;
  
  // In case of 1 pair, copy it into concrete domain array
  if ( _num_pairs == 1 )
  {
  	_concrete_domain [ 0 ] = _lower_bound;
  	_concrete_domain [ 1 ] = _upper_bound;
  }
}//set_domain

unsigned int
CudaConcreteDomainList::size () const 
{
	return _domain_size;
}//size

int 
CudaConcreteDomainList::get_elements_in_bound ( int pair_idx )
{
	if ( _concrete_domain[ 2 * pair_idx + 1 ] >= 0 && _concrete_domain[ 2 * pair_idx ] >= 0 )
	{
		return _concrete_domain[ 2 * pair_idx + 1 ] - _concrete_domain[ 2 * pair_idx ] + 1;
	}
	if ( _concrete_domain[ 2 * pair_idx + 1 ] >= 0 &&  _concrete_domain[ 2 * pair_idx ] < 0 )
	{
		return _concrete_domain[ 2 * pair_idx + 1 ] + abs ( _concrete_domain[ 2 * pair_idx ] ) + 1;
	}
	
	return abs ( _concrete_domain[ 2 * pair_idx ] ) - abs ( _concrete_domain[ 2 * pair_idx + 1 ] ) + 1;
}//get_elements_in_bound

std::size_t 
CudaConcreteDomainList::get_size_from_values ( int min, int max )
{
	if ( min >= 0 && max >= 0 )
    {
    	return (max - min) + 1;
	}
	else if ( min < 0 && max > 0 )
	{
		return max + abs ( min ) + 1;
	}
	else
	{// min < 0 /\ max < 0
		return abs ( min ) - abs ( max ) + 1;
	}
}//get_size_from_values

std::size_t 
CudaConcreteDomainList::get_size_from_bounds_btw ( int lower_pair, int upper_pair )
{
	// Sanity check
	assert ( lower_pair >= 0 && upper_pair < _num_pairs );
	
	std::size_t sum_elements = 0;
    for ( int pair_idx = lower_pair + 1; pair_idx <= upper_pair; pair_idx++ ) 
    {
    	if ( _concrete_domain[ 2 * (pair_idx - 1) + 1 ] >= 0 && _concrete_domain[ 2 * pair_idx ] >= 0 )
    	{
    		sum_elements +=
      		_concrete_domain[ 2 * pair_idx           ] -
      		_concrete_domain[ 2 * (pair_idx - 1) + 1 ] - 1;
    	}
    	else if ( _concrete_domain[ 2 * (pair_idx - 1) + 1 ] < 0 && _concrete_domain[ 2 * pair_idx ] >= 0 )
    	{
    		sum_elements +=
      	          _concrete_domain[ 2 * pair_idx           ]   +
      		abs ( _concrete_domain[ 2 * (pair_idx - 1) + 1 ] ) - 1;
    	}
      	else
      	{
      		sum_elements +=
      		abs ( _concrete_domain[ 2 * (pair_idx - 1) + 1 ] ) -
      	    abs ( _concrete_domain[ 2 * pair_idx           ] ) - 1;
      		
      	}
    }//pair_idx
    
    return sum_elements;
}//get_size_from_bounds_btw

int
CudaConcreteDomainList::find_pair ( int val ) const 
{
	if ( _num_pairs == 0 ) return -1;
  
  	// Scan the pairs to check for val
  	for ( int pair_idx = 0; pair_idx < _num_pairs; pair_idx++ ) 
  	{
    	if ( val >= _concrete_domain [ 2 * pair_idx     ] &&
        	 val <= _concrete_domain [ 2 * pair_idx + 1 ]) 
        {
      		return  pair_idx;
    	}
  	}
	return -1;
}//find_pair

int
CudaConcreteDomainList::find_prev_pair ( int val ) const 
{
	if ( _num_pairs == 0 ) return -1;
  
  	for ( int pair_idx = 0; pair_idx < _num_pairs - 1; pair_idx++ ) 
  	{
    	if ( (_concrete_domain[ 2 * pair_idx + 1   ] < val) &&
         	 (_concrete_domain[ 2 * (pair_idx + 1) ] > val) ) 
    	{
      		return  pair_idx;
    	}
  	}
  
  	// Check last pair
  	if ( _concrete_domain[ 2 * (_num_pairs - 1) + 1 ] < val )
  	{
    	return _num_pairs - 1;
  	}
  	
  	return -1;
}//find_prev_pair

int
CudaConcreteDomainList::find_next_pair ( int val ) const 
{ 
	if ( _num_pairs == 0 ) return -1;
  
  	// Check first pair
  	if ( _concrete_domain[ 0 ] > val ) return 0;
  
  	for ( int pair_idx = 1; pair_idx < _num_pairs; pair_idx++ ) 
  	{
    	if ( (_concrete_domain[ 2 * pair_idx           ] > val) &&
        	 (_concrete_domain[ 2 * (pair_idx - 1) + 1 ] < val) ) 
        {
      		return  pair_idx;
    	}
  	}
  	return -1;
}//find_prev_pair

void
CudaConcreteDomainList::shrink ( int min, int max ) 
{
	// Empty domain if not consistent
  	if ( max < min ) 
  	{
    	flush_domain ();
    	return;
  	}
  	std::cout << "CudaConcreteDomainList::shrink: " << _lower_bound << " " << _upper_bound << 
  	" " << min << " " << max << std::endl;
  	// Sanity check for min/max values
  	if ( min == _lower_bound && max == _upper_bound ) return;
  
  	// Return if no chages in the domain
  	if ( (min <= _lower_bound) && (max >= _upper_bound) ) return;
  
  	// Set min/max w.r.t. the current bounds
  	if ( (min < _lower_bound) && (max < _upper_bound) ) 
  	{
    	min = _lower_bound;
  	}
  
  	if ( (min > _lower_bound) && (max > _upper_bound) ) 
  	{
    	max = _upper_bound;
  	}

  	/*
   	 * There are _num_pairs pairs of contiguous elements.
   	 * Four cases to consider for both the new lower/upper bounds:
   	 * 1) new bound is equal to a lower bound
   	 * 2) new bound is equal to an upper bound
   	 * 3) new bound is within a pair {lower bound, upper bound}
   	 * 4) new bound is between to pair of bounds
   	 */
   	 
  	// Size (num of elements) of the current pair
  	int pair_size{};
  
  	// Prev (next) sum of vals w.r.t. min (max)
  	int  sum_prev_elements{};
  	int  sum_next_elements{};
  
  	// Index of the pair containing min (max)
  	int pair_with_min = -1;
  	int pair_with_max = -1;
  
  	/*
   	 * State whether the sum of vals < min (vals > max)
   	 * is calculated (needed for calc. domain's size)
   	 */
 	bool sum_prev_vals_complete = false;
  	bool sum_next_vals_complete = false;
  
  	// Number of valid pairs
  	int num_pairs_in_domain{};
  	for ( int pair_idx = 0; pair_idx < _num_pairs; pair_idx++ ) 
  	{    
    	// min is within this current pair of bounds
    	if ( (min >= _concrete_domain[ 2 * pair_idx     ]) &&
        	 (min <= _concrete_domain[ 2 * pair_idx + 1 ]) ) 
        {
      		pair_with_min          = pair_idx;
      		sum_prev_vals_complete = true;
      
      		// Add the elements < min from this current subset
      		if ( (min <  0 && _concrete_domain[ 2 * pair_idx ] <  0) ||  
      		     (min >= 0 && _concrete_domain[ 2 * pair_idx ] >= 0) )
      		{
      			sum_prev_elements += abs ( abs ( min ) - abs ( _concrete_domain[ 2 * pair_idx ] ) );
      		}
      		else 
      		{// _concrete_domain[ 2 * pair_idx ] < 0 && min >= 0 	
      			sum_prev_elements += (min + abs ( _concrete_domain[ 2 * pair_idx ] ));
      		}
      
      		// Update min for this pair of bounds
      		_concrete_domain[ 2 * pair_idx ] = min;
    	}
    
    	// Add this pair if min has not been found yet (i.e., elements < min)
    	if ( !sum_prev_vals_complete ) 
    	{
      		sum_prev_elements += pair_size;
    	}
    
    	/*
     	 * Add 1 to the number of pairs between min and max
     	 * when min has been found but not max yet
     	 */
    	if ( sum_prev_vals_complete && !sum_next_vals_complete ) 
    	{
      		num_pairs_in_domain += 1;
    	}
    
    	// Calculate the size of the current subset of domain
    	pair_size = get_elements_in_bound ( pair_idx );

    	// Add this pair if max has been found (i.e., elements > max)
    	if ( sum_next_vals_complete ) 
    	{
      		sum_next_elements += pair_size;
    	}
    
    	// max is within this pair of bounds
    	if ( (max >= _concrete_domain[ 2 * pair_idx     ]) &&
         	 (max <= _concrete_domain[ 2 * pair_idx + 1 ]) ) 
        {
      		pair_with_max          = pair_idx;
      		sum_next_vals_complete = true;
      
      		// Add the elements > max from this current subset
      		if ( (max <  0 && _concrete_domain[ 2 * pair_idx + 1] <  0) ||  
      		     (max >= 0 && _concrete_domain[ 2 * pair_idx + 1] >= 0) )
      		{
      			sum_next_elements += abs ( abs ( _concrete_domain[ 2 * pair_idx + 1 ] ) - abs ( max ) );
      		}
      		else 
      		{// _concrete_domain[ 2 * pair_idx + 1 ] > 0 && max <= 0 	
      			sum_next_elements += (_concrete_domain[ 2 * pair_idx + 1 ] + abs ( max ));
      		}
      
      		// Update max for this pair of bounds
      		_concrete_domain[ 2 * pair_idx + 1 ] = max;
    	}

    	/*
    	 * Check when min (max) is between to pairs, e.g.,
     	 * min = 7 in {3, 6} {10, 14}.
     	 */
    	if ( pair_idx < _num_pairs - 1 ) 
    	{
      		// Pair with min: next pair
      		if ( (min > _concrete_domain[ 2 * pair_idx + 1   ]) &&
           		 (min < _concrete_domain[ 2 * (pair_idx + 1) ]) ) 
           	{
        		pair_with_min = pair_idx + 1;
        		sum_prev_vals_complete = true;
      		}
      
      		// Pair with max: current pair
      		if ( (max > _concrete_domain[ 2 * pair_idx + 1   ]) &&
           		 (max < _concrete_domain[ 2 * (pair_idx + 1) ]) ) 
           	{
        		pair_with_max          = pair_idx;
        		sum_next_vals_complete = true;
      		}
    	}
  	}//pair
  
	/*
   	 * Fail (empty domain) if no pairs are valid.
   	 * For example: min = 8, max = 10 in
   	 * {2, 7}, {15, 20}.
   	 */
  	if ( num_pairs_in_domain == 0 ) 
  	{
    	flush_domain ();
    	return;
  	}
  	
  	// Set new number of valid pairs
  	_num_pairs = num_pairs_in_domain;
  
  	// Reduce total size
  	_domain_size -= (sum_prev_elements + sum_next_elements);
  	
  	/*
   	 * Translate the pairs on the left:
   	 * {1, 4} {6, 10} {22, 25} (min = 8, max = 23) ->
   	 * {8, 10}, {22, 23}.
   	 */
  	if ( pair_with_min > 0 ) 
  	{
    	memcpy( _concrete_domain,
            	&_concrete_domain[ 2 * pair_with_min ],
            	2 * _num_pairs * sizeof( int ) );
  	}
  
  	// Update bounds
  	_lower_bound = _concrete_domain[ 0 ];
  	_upper_bound = _concrete_domain[ 2 * pair_with_max + 1 ];
}//shrink

void
CudaConcreteDomainList::subtract ( int value ) 
{
	if ( value == _lower_bound ) 
	{
    	in_min ( ++value );
    	return;
  	}
  
  	if ( value == _upper_bound ) 
  	{
    	in_max ( --value );
    	return;
  	}
}//subtract

void
CudaConcreteDomainList::in_min ( int min ) 
{
	if ( min <= _lower_bound ) return;
  	shrink ( min, _upper_bound );
}//in_min

void
CudaConcreteDomainList::in_max ( int max ) 
{
	if ( max >= _upper_bound ) return;
  	shrink ( _lower_bound, max );
}//in_min

void
CudaConcreteDomainList::add ( int value ) 
{
	// Sanity check
  	if ( value < _init_lower_bound ) value = _init_lower_bound;
  	if ( value > _init_upper_bound ) value = _init_upper_bound;
  
  	// Return if the value is already set to 1
  	if ( contains( value ) ) return;
  
  	add ( value, value );
}//add

void
CudaConcreteDomainList::add ( int min, int max ) 
{  
	// Sanity check
  	if ( max < min ) return;
  
  	// Find the pair containing min (max)
  	int lower_pair = find_pair ( min );
  	int upper_pair = find_pair ( max );
  	
  	/*
     * Example:
   	 * min = 13, max = 35
   	 * {1, 4}, {10, 15}, {20, 25}, {30, 37}, {40, 50} ->
   	 * {1, 4}, {10, 37}, {40, 50}
   	 * where
   	 * _num_pairs       = 5
   	 * lower_pair       = 1
   	 * upper_pair       = 3
   	 * num_pairs_within = 1 (i.e., {20, 25})
	 * num_pairs_right  = 1 (i.e., {40, 50})
   	 */
  	if ( (lower_pair >= 0) && (upper_pair >= 0) ) 
  	{
    	// No action if min/max are in the same pair
    	if ( lower_pair == upper_pair ) return;
    
    	int num_pairs_within = upper_pair - lower_pair - 1;
    
    	// Find the number of elements to add
    	std::size_t sum_elements = get_size_from_bounds_btw ( lower_pair, upper_pair );

    	// Set new domain's size
    	_domain_size += sum_elements;
    
    	// Reset pairs and move to the left pairs > upper_pair
    	int value_to_set    = _concrete_domain [ 2 * upper_pair + 1 ];
    	int num_pairs_right = _num_pairs - upper_pair - 1;
    	if ( num_pairs_right )
    	{
      		memcpy( &_concrete_domain [ 2 * (lower_pair + 1) ],
              	    &_concrete_domain [ 2 * (upper_pair + 1) ] ,
              	    2 * num_pairs_right * sizeof( int ) );
    	}
    	
    	_concrete_domain [ 2 * lower_pair + 1 ] = value_to_set;
    
    	// Set new total number of pairs
    	_num_pairs -= (num_pairs_within + 1);
    	return;
  	}
  
	/*
   	 * Example:
   	 * min = 13, max = 38
   	 * {1, 4}, {10, 15}, {20, 25}, {30, 37}, {40, 50} ->
   	 * {1, 4}, {10, 38}, {40, 50}
   	 * where
   	 * _num_pairs       = 5
   	 * prev_pair_idx    = 3
   	 * lower_pair       = 1
   	 * num_pairs_within = 1 (i.e., {20, 25})
   	 */
  	if ( lower_pair >= 0 ) 
  	{
    	// Find prev pair and add elements to it
    	int prev_pair_idx = find_prev_pair ( max );
    
    	// Find the number of elements to add
    	std::size_t sum_elements = get_size_from_bounds_btw ( lower_pair, prev_pair_idx - 1 );
    
    	sum_elements += max - _concrete_domain[ 2 * prev_pair_idx + 1 ];
    
    	// Set new domain size
    	_domain_size += sum_elements;
 
    	// Reset pairs and move to the left pairs > upper_pair
    	int num_pairs_within = prev_pair_idx - lower_pair - 1;
    	int num_pairs_right = _num_pairs - prev_pair_idx - 1;
    
    	if ( num_pairs_right )
    	{
      		memcpy( &_concrete_domain [ 2 * (lower_pair + 1)  ],
              		&_concrete_domain [ 2 * (prev_pair_idx + 1) ] ,
              		2 * num_pairs_right * sizeof( int ) );
        }
    
    	_concrete_domain [ 2 * lower_pair + 1 ] = max;
    
    	// Set new total number of pairs
    	_num_pairs -= (num_pairs_within + 1);
    	return;
  	}
  
  	/*
   	 * Example:
   	 * min = 18, max = 36
   	 * {1, 4}, {10, 15}, {20, 25}, {30, 37}, {40, 50} ->
   	 * {1, 4}, {10, 15}, {18, 37}, {40, 50}
   	 * where
   	 * _num_pairs       = 5
   	 * next_pair_idx    = 2
   	 * upper_pair       = 3
   	 * num_pairs_right  = 1
   	 * num_pairs_within = 0
   	 */
  	if ( upper_pair >= 0 ) 
  	{
    	int next_pair_idx = find_next_pair ( min );
    
    	// Find the number of elements to add
    	std::size_t sum_elements = get_size_from_bounds_btw ( next_pair_idx, upper_pair );
    
    	sum_elements += _concrete_domain[ 2 * next_pair_idx ] - min;
    
    	// Set new domain size
    	_domain_size += sum_elements;
    
    	// Reset pairs and move to the left pairs > upper_pair
    	int num_pairs_within = upper_pair - next_pair_idx - 1;
    	int num_pairs_right = _num_pairs - upper_pair - 1;
    
    	// Update bounds
    	_concrete_domain [ 2 * next_pair_idx     ] = min;
    	_concrete_domain [ 2 * next_pair_idx + 1 ] = _concrete_domain[ 2 * upper_pair + 1 ];
    
    	// Copy only if there are pairs on the right to copy
    	if ( num_pairs_right > 0 )
    	{
      		memcpy( &_concrete_domain [ 2 * (next_pair_idx + 1) ],
             	    &_concrete_domain [ 2 * (upper_pair + 1)    ] ,
             		2 * num_pairs_right * sizeof( int ) );
        }
    
    	// Set new total number of pairs
    	_num_pairs -= (num_pairs_within + 1);
   	 	return;
  	}
  
  	/*
   	 * Example:
   	 * min = 18, max = 38
   	 * {1, 4}, {10, 15}, {20, 25}, {30, 37}, {40, 50} ->
   	 * {1, 4}, {10, 15}, {18, 38}, {40, 50}
   	 * where
   	 * _num_pairs       = 5
   	 * prev_pair_idx    = 3
   	 * next_pair_idx    = 2
   	 * num_pairs_within = 1
   	 * num_pairs_right  = 1
   	*/
  	if ( lower_pair < 0 && upper_pair < 0 ) 
  	{
    	int next_pair_idx = find_next_pair ( min );
    	int prev_pair_idx = find_next_pair ( max );
  	
    	// If prev/next < 0: no elements -> add current pair and exit
    	if ( prev_pair_idx < 0 && next_pair_idx < 0 ) 
    	{
      		_concrete_domain[ 0 ] = min;
      		_concrete_domain[ 1 ] = max;
      		_domain_size = get_size_from_values ( min, max );
			
      		// Update bounds
      		if ( min < _lower_bound ) _lower_bound = min;
      		if ( max > _upper_bound ) _upper_bound = max;
      		_num_pairs += 1;
      		return;
    	}
    
    	/* Add a new pair
     	 * min = 18, max = 19
     	 * {1, 4}, {10, 15}, {20, 25}, {30, 37}, {40, 50} ->
     	 * {1, 4}, {10, 15}, {18, 19}, {20, 25}, {30, 37}, {40, 50}
     	 * or
     	 * {1, 4} ->
     	 * {1, 4}, {18, 19}
     	 */
    	if ( (next_pair_idx > prev_pair_idx) || ((next_pair_idx < 0) && ( prev_pair_idx >= 0 )) ) 
    	{
      		if ( _num_pairs + 1 > _max_allowed_pairs ) 
      		{
        		LogMsg.error( _dbg + "Can't add another pair", __FILE__, __LINE__ );
        		set_empty ();
        		return;
      		}
      
      		// Shift domain on the right
      		int num_pairs_right = _num_pairs - prev_pair_idx - 1;
      		if ( num_pairs_right > 0 )
      		{
        		memcpy( &_concrete_domain[ 2 * (next_pair_idx + 1) ],
               			&_concrete_domain[ 2 * (prev_pair_idx + 1) ],
               			2 * num_pairs_right * sizeof( int ) );
            }
      
      		if ( next_pair_idx < 0 ) next_pair_idx = prev_pair_idx + 1;
      		_concrete_domain[ 2 * next_pair_idx     ] = min;
      		_concrete_domain[ 2 * next_pair_idx + 1 ] = max;
      
      		_num_pairs++;
      		_domain_size += get_size_from_values ( min, max );
      		return;
    	}
    
    	// Find the number of elements to add
    	std::size_t sum_elements = get_size_from_bounds_btw ( next_pair_idx, upper_pair );
    	sum_elements += get_size_from_values ( min, _concrete_domain[ 2 * next_pair_idx ] ) - 1;
    	sum_elements += get_size_from_values ( _concrete_domain[ 2 * prev_pair_idx ], max ) - 1;
    
    	// Set new domain size
    	_domain_size += sum_elements;
    
    	// Reset pairs and move to the left pairs > upper_pair
    	int num_pairs_within = prev_pair_idx - next_pair_idx;
    	int num_pairs_right = _num_pairs - prev_pair_idx - 1;
    
    	// Copy only if there are pairs on the right to copy
    	if ( num_pairs_right )
    	{
      		memcpy( &_concrete_domain [ 2 * (next_pair_idx + 1) ],
              		&_concrete_domain [ 2 * (prev_pair_idx + 1) ] ,
              		2 * num_pairs_right * sizeof( int ) );
        }
    
    	_concrete_domain [ 2 * next_pair_idx     ] = min;
    	_concrete_domain [ 2 * next_pair_idx + 1 ] = max;
    
    	// Fix total num pairs
    	_num_pairs -= num_pairs_within;
  	}
  
  	// Update bounds
  	if ( min < _lower_bound ) _lower_bound = min;
  	if ( max > _upper_bound ) _upper_bound = max;
}//add

bool
CudaConcreteDomainList::contains ( int val ) const 
{
	return  ( find_pair( val ) != -1 );
}//contains

bool
CudaConcreteDomainList::is_singleton () const 
{
	if ( _num_pairs == 1 ) 
	{
    	return size() == 1;
  	}
  	return  false;
}//is_singleton

int
CudaConcreteDomainList::get_singleton () const 
{
	if ( !is_singleton() ) 
	{
    	throw  NvdException ( ( _dbg + "Domain not singleton" ).c_str() );
  	}
  
  	return _lower_bound;
}//get_singleton

int
CudaConcreteDomainList::get_id_representation () const 
{
	return _num_pairs;
}//get_id_representation

void
CudaConcreteDomainList::print () const 
{
	for ( int pair_idx = 0; pair_idx < _num_pairs; pair_idx++ ) 
	{
    	std::cout <<
    	"{" <<
    	_concrete_domain [ 2 * pair_idx     ] <<
    	", " <<
    	_concrete_domain [ 2 * pair_idx + 1 ] <<
    	"} ";
  	}
}//print



