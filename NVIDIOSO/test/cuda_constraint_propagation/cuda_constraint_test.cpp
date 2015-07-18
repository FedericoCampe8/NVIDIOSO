#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <string>
#include <queue>
#include <utility>
#include <set>
#include <string.h>
#include <math.h>

#define nullptr NULL
#define uint unsigned int

#define MAX_SIZE 256
#define STANDARD_DOM MAX_SIZE / (8*sizeof(int))

// EVENTS ON DOMAINS
#define NOP_EVT 0
#define SNG_EVT 1
#define BND_EVT 2
#define MIN_EVT 3
#define MAX_EVT 4
#define CHG_EVT 5
#define FAL_EVT 6

// STANDARD REPRESENTATION
#define BIT_REP 0
#define BND_REP 1

// BOOLEAN DOMAIN REPRESENTATION
#define BOL_EVT 7
#define BOL_F   0
#define BOL_T   1
#define BOL_U   2
// INDEX ON DOMAINS
#define EVT 0
#define REP 1
#define LB  2
#define UB  3
#define DSZ 4
#define BIT 5

// INDEX FOR BOOLEAN STATUS REPRESENTATION
#define ST  1

// VARIABLES (DOMAINS) TYPE
#define MIXED_DOM    1
#define BOOLEAN_DOM  2

// STATUS
#define STATUS(x, y) (_status[(x)][(y)])
#define GET_VAR_EVT(x)  _status[(x)][EVT]
#define GET_VAR_REP(x)  _status[(x)][REP]
#define GET_VAR_LB(x)   _status[(x)][LB]
#define GET_VAR_UB(x)   _status[(x)][UB]
#define GET_VAR_DSZ(x)  _status[(x)][DSZ]

// DEVELOPER "FRIENDLY" ALIAS
#define NUM_ARGS _args_size
#define NUM_VARS _scope_size
#define ARGS     _args
#define VARS     _vars
#define C_ARG    _args[0]
#define LAST_VAR_IDX ((_scope_size > 0) ? (_scope_size - 1) : 0)
#define LAST_ARG_IDX ((_args_size > 0) ? (_args_size - 1) : 0)
#define X_VAR    0
#define Y_VAR    1
#define Z_VAR    2

#define BITS_IN_CHUNK 32
#define NUM_CHUNKS 8

using namespace std;

uint* domain_states;

int _scope_size;
int _args_size;
int* _vars;
int* _args;
uint** _status;
uint** _temp_status;

void print_bit ( uint v )
{
    uint m = 1;
    for ( int i = 8 * sizeof ( int )-1; i >= 0; i-- )
    {
        if ( v & (m << i) ) cout << "1";
        else                cout << "0";
    }
    cout << endl;
}//print_bit

void init ()
{
    _scope_size = 4;
    _args_size  = 5;
    _vars = new int [ _scope_size ];
    _args = new int [ _args_size  ];
    
    // Domain states
    uint val = 0;
    for ( int i = 1; i <= 20; i++ )
    {
        uint mask = 1;
        mask = mask << i;
        val |= mask;
    }

    domain_states = new uint [ _scope_size * 13 ];
    for ( int i = 0; i < _scope_size; i++ )
    {
        int idx = 13 * i;
        domain_states[ idx + EVT ] = NOP_EVT;
        domain_states[ idx + REP ] = BIT_REP;
        domain_states[ idx + LB ]  = 1;
        domain_states[ idx + UB ]  = 20;
        domain_states[ idx + DSZ ] = 20;
        for ( int ii = 0; ii < 8; ii++ )
            domain_states[ idx + BIT + ii ] = 0;
        domain_states[ idx + BIT + 7 ] = val;
    }
    
    for ( int i = 0; i < _scope_size; i++ )
    {
        _vars [ i ] = i;
    }
    for ( int i = 0; i < _args_size; i++ )
    {
        _args [ i ] = 1000;
    }
    _status = (uint**) malloc ( _scope_size * sizeof (uint*) );
    _temp_status = (uint**) malloc ( _scope_size * sizeof (uint*) );

    for ( int i = 0; i < _scope_size; i++ )
    {
        int idx = 13 * i;
        _status[ i ] = &domain_states [ idx ];
    }
}//init

void clean ()
{
    delete [] _vars;
    delete [] _args;
    delete [] domain_states;
}//clean

 bool all_ground ()
{
    for ( int i = 0; i < _scope_size; i++ )
        if ( _status[ i ][ EVT ] != SNG_EVT )
            return false;
    return true;
}//all_ground

bool
only_one_not_ground ()
{
    int not_grd = 0;
    for ( int i = 0; i < _scope_size; i++ )
    {
        if ( _status[ i ][ EVT ] != SNG_EVT )
        {
            not_grd++;
            if ( not_grd > 1 )
                return false;
        }
    }
    return true;
}//only_one_not_ground

int
get_not_ground ()
{
    for ( int i = 0; i < _scope_size; i++ )
        if ( _status[ i ][ EVT ] != SNG_EVT )
            return i;
    return -1;
}//get_not_ground

bool
is_singleton ( int var )
{
    return (_status[ var ][ EVT ] == SNG_EVT);
}//is_singleton

bool
is_ground ( int var )
{
    return _status[ var ][ EVT ] == SNG_EVT;
}//is_ground

bool
contains ( int var, int val )
{
    if ( _status[ var ][ EVT ] == BOL_EVT )
    {
        int val_in = _status[ var ][ ST ];
        if ( val_in == BOL_U )
        {
            return (val >= 0 && val <= 1);
        }
        return val == val_in;
    }

    int chunk = val / BITS_IN_CHUNK;
    chunk = BIT + NUM_CHUNKS - 1 - chunk;
    return ( (_status [ var ][ chunk ] & (1 << (val % BITS_IN_CHUNK))) != 0 );
}//contains

void
subtract ( int var, int val )
{
    
    // Boolean domain
    if ( _status[ var ][ EVT ] == BOL_EVT )
    {
        if ( val < 0 || val > 1 ) return;

        int b_val = _status[ var ][ ST ];
        if ( b_val == val )
        {// b_val is 0 or 1 and status is 0 or 1 -> fail
            _status[ var ][ EVT ] = FAL_EVT;
        }
        else
        {// b_val is BOL_U (undef) -> remove one val and set singleton
            _status[ var ][ ST ]   = val;
            _status[ var ][ EVT ]  = SNG_EVT;
        }
    }
    else
    {
        // @note Only one pair of bounds is allowed
        int lower_bound = _status [ var ][ LB ];
        int upper_bound = _status [ var ][ UB ];
        if ( val < lower_bound || val > upper_bound ) return;
        if ( _status [ var ][ REP ] == 0 )
        {// Bitmap representation for domain
            /*
             * Find the chunk and the position of the bit within the chunk.
             * @note: chunks are stored in reversed/LSB order.
             *   For example: {0, 63} is stored as
             *        | 0 | 0 | 0 | 0 | 0 | 0 | 63...32 | 31...0 |
             */
            int chunk = val / BITS_IN_CHUNK;
            chunk = BIT + NUM_CHUNKS - 1 - chunk;
            uint val_chunk = _status[ var ][ chunk ];
            uint val_clear = val % BITS_IN_CHUNK;

            // Check is the bit is already unsed
            if ( !((val_chunk & ( 1 << val_clear )) != 0) ) return;
            _status[ var ][ chunk ] = val_chunk & (~(1 << val_clear ));

            int domain_size = _status[ var ][ DSZ ] - 1;
            if ( domain_size <= 0 )
            {// Failed event
                _status[ var ][ EVT ] = FAL_EVT;
                return;
            }
            if ( domain_size == 1 )
            {// Singleton event
                _status[ var ][ DSZ ] = 1;

                // Lower bound increased
                if ( lower_bound == val )
                {
                    _status[ var ][ LB ]  = upper_bound;
                    _status[ var ][ EVT ] = SNG_EVT;
                    return;
                }
                // Upper bound decreased
                _status[ var ][ UB ]  = lower_bound;
                _status[ var ][ EVT ] = SNG_EVT;
                return;
            }

            // Lower bound increased
            if ( lower_bound == val )
            {
                while ( true )
                {
                    lower_bound++;
                    if ( contains ( var, lower_bound ) )
                    {
                        _status[ var ][ LB ]  = lower_bound;
                        break;
                    }
                }
                _status[ var ][ EVT ] = MIN_EVT;
            }
            else if ( upper_bound == val )
            {
                while ( true )
                {
                    upper_bound--;
                    if ( contains ( var, upper_bound ) )
                    {
                        _status[ var ][ UB ]  = upper_bound;
                        break;
                    }
                }
                _status[ var ][ EVT ] = MAX_EVT;
            }
            else
            {
                _status[ var ][ EVT ] = CHG_EVT;
            }
            _status[ var ][ DSZ ] = domain_size;
        }
        else
        {// Pair of bounds
            if ( val > lower_bound && val < upper_bound ) return;
            if ( lower_bound == val )
            {
                if ( upper_bound == val )
                {
                    _status[ var ][ EVT ] = FAL_EVT;
                    return;
                }
                if ( upper_bound == val+1 )
                {
                    _status[ var ][ LB ]  = upper_bound;
                    _status[ var ][ EVT ] = SNG_EVT;
                    return;
                }

                _status[ var ][ LB ]  = lower_bound + 1;
                _status[ var ][ EVT ] = BND_EVT;
                return;
            }
            else if ( upper_bound == val )
            {
                if ( lower_bound == val )
                {
                    _status[ var ][ EVT ] = FAL_EVT;
                    return;
                }
                if ( lower_bound == val-1 )
                {
                    _status[ var ][ UB ]  = lower_bound;
                    _status[ var ][ EVT ] = SNG_EVT;
                    return;
                }

                _status[ var ][ UB ]  = upper_bound - 1;
                _status[ var ][ EVT ] = BND_EVT;
                return;
            }
        }
    }
}//subtract

int
get_min ( int var )
{
    if ( _status[ var ][ EVT ] == BOL_EVT )
    {
        int val = _status[ var ][ ST ];
        if ( val == BOL_U )
        {
            return BOL_F;
        }
        return val;
    }

    // Standard domain representation
    return _status[ var ][ LB ];
}//get_min

int
get_max ( int var )
{
    if ( _status[ var ][ EVT ] == BOL_EVT )
    {
        int val = _status[ var ][ ST ];
        if ( val == BOL_U )
        {
            return BOL_T;
        }
        return val;
    }

    // Standard domain representation
    return _status[ var ][ UB ];
}//get_max

int
get_sum_ground ()
{
    int product = 0;
    for ( int idx = 0; idx < NUM_VARS; idx++ )
    {
        if ( is_singleton ( idx ) )
        {
            product += ARGS[idx] * get_min ( idx );
        }
    }
    return product;
}//get_sum_ground

 bool
is_empty_var ( int var )
{
    return ( _status[ var ][ EVT ] == FAL_EVT );
}//is_empty_var

void
clear_bits_i_through_0 ( uint& val, int i )
{
    int mask = ~((1 << (i+1)) - 1);
    val = val & mask;
}//clear_bits_i_through_0

void
clear_bits_MSB_through_i ( uint& val, int i )
{
    val = val & ( ( 1 << i ) - 1 );
}//clear_bits_MSB_through_i

int
num_1bit ( uint n )
{
    int c = 0;
    for ( c = 0; n; c++ )
        n &= n - 1;
    return c;
}//num_1bit


void
 shrink ( int var, int smin, int smax )
{
    // Boolean domain
    if ( _status[ var ][ EVT ] == BOL_EVT )
    {
        if ( smin < 0 || smin > 1 || smax < 0 || smax > 1 || smax < smin ) return;
        if ( smin + smax == 1 )
        {// b_val is 0 or 1 and status is 0 or 1 -> fail
            _status[ var ][ EVT ] = FAL_EVT;
        }
        if ( smin == smax )
        {
            subtract ( var, smin );
        }
    }
    else
    {
        // @note Only one pair of bounds is allowed
        int lower_bound = _status [ var ][ LB ];
        int upper_bound = _status [ var ][ UB ];
        if ( smin <= lower_bound && smax >= upper_bound ) return;
        if ( smin > smax )
        {
            _status[ var ][ EVT ] = FAL_EVT;
            return;
        }

        smin = (smin > lower_bound) ? smin : lower_bound;
        smax = (smax < upper_bound) ? smax : upper_bound;
        
        if ( smin == smax )
        {
            _status[ var ][ DSZ ] = 1;
            _status[ var ][ LB ]  = smin;
            _status[ var ][ UB ]  = smin;
            _status[ var ][ EVT ] = SNG_EVT;
            return;
        }
        if ( _status [ var ][ REP ] == 0 )
        {// Bitmap representation for domain
            
            int chunk_min = smin / BITS_IN_CHUNK;
            chunk_min = BIT + NUM_CHUNKS - 1 - chunk_min;
            int chunk_max = smax / BITS_IN_CHUNK;
            chunk_max = BIT + NUM_CHUNKS - 1 - chunk_max;
            for ( int i = BIT; i < chunk_min; i++ )                  _status [ var ][ i ] = 0;
            for ( int i = BIT + NUM_CHUNKS - 1; i > chunk_max; i-- ) _status [ var ][ i ] = 0;
            clear_bits_i_through_0   ( _status [ var ][ chunk_min ], (smin % BITS_IN_CHUNK) - 1 );
            clear_bits_MSB_through_i ( _status [ var ][ chunk_max ], (smax % BITS_IN_CHUNK) + 1 );

            int num_bits = 0;
            for ( int i = chunk_min; i <= chunk_max; i++ )
                num_bits += num_1bit ( (uint) _status [ var ][ i ] );

            if ( num_bits == 0 )
            {
                _status[ var ][ EVT ] = FAL_EVT;
                return;
            }
            if ( num_bits == 1 )
            {
                _status[ var ][ DSZ ] = 1;
                _status[ var ][ LB ]  = smin;
                _status[ var ][ UB ]  = smin;
                _status[ var ][ EVT ] = SNG_EVT;
                return;
            }

            _status[ var ][ DSZ ] = num_bits;
            _status[ var ][ LB ]  = (smin < lower_bound) ? lower_bound : smin;
            _status[ var ][ UB ]  = (smax > upper_bound) ? upper_bound : smax;
            while ( true )
            {
                if ( contains ( var, lower_bound ) )
                {
                    _status[ var ][ LB ]  = lower_bound;
                    break;
                }
                lower_bound++;
            }
            while ( true )
            {
                if ( contains ( var, upper_bound ) )
                {
                    _status[ var ][ UB ]  = upper_bound;
                    break;
                }
                upper_bound--;
            }

            _status[ var ][ LB ]  = lower_bound;
            _status[ var ][ UB ]  = upper_bound;
            _status[ var ][ EVT ] = CHG_EVT;
        }
        else
        {// Pair of bounds
            
            if ( smin > lower_bound && smax > upper_bound )
            {
                int cnt = smin - lower_bound;
                _status[ var ][ LB ] = smin;
                _status[ var ][ DSZ ] -= cnt;
                _status[ var ][ EVT ] = BND_EVT;
                return;
            }
            if ( smin < lower_bound && smax < upper_bound )
            {
                int cnt = upper_bound - smax;
                _status[ var ][ UB ] = smax;
                _status[ var ][ DSZ ] -= cnt;
                _status[ var ][ EVT ] = BND_EVT;
                return;
            }
            int cnt = (smin - lower_bound) + (upper_bound - smax);
            _status[ var ][ LB ] = smin;
            _status[ var ][ UB ] = smax;
            _status[ var ][ DSZ ] -= cnt;
            _status[ var ][ EVT ] = BND_EVT;
        }
    }
}//shrink    

void print ( int var = -1 )
{
    int start    = 0;
    int num_vars = _scope_size;
    if ( var >= 0 && var < _scope_size )
    {
        start    = var;
        num_vars = start + 1;
    }
    for ( int i = start; i < num_vars; i++ )
    {
        cout << "V_" << i << ": ";
        for ( int j = 0; j < 5; j++ )
        {
            cout << _status [ i ][ j ] << " ";
        }
        cout << endl;
    }
}//print

void
move_status_to_shared (  uint * shared_ptr, int dom_size )
{
    if ( shared_ptr == nullptr ) return;
    if ( dom_size == MIXED_DOM )
    {
        int d_size, idx = 0;
        for ( int i = 0; i < _scope_size; i++ )
        {
            _temp_status [ i ] = _status [ i ];
            _status [ i ]      = &shared_ptr [ idx ];
            if ( _status [ i ][ EVT ] == BOL_EVT )
            {
                d_size = 2;
            }
            else
            {
                d_size = STANDARD_DOM + 5;
            }
            
            for ( int j = 0; j < d_size; j++ )
            {
                //cout << _temp_status [ i ][ j ] << " ";
                shared_ptr [ idx++ ] = _temp_status [ i ][ j ];
            }
            //cout << endl;
        }
    }
}//move_status_to_shared

void
move_status_from_shared ( uint * shared_ptr, int dom_size )
{
    if ( shared_ptr == nullptr ) return;
    if ( dom_size == MIXED_DOM )
    {
        int d_size, idx = 0;
        for ( int i = 0; i < _scope_size; i++ )
        {
            if ( _status [ i ][ EVT ] == BOL_EVT )
            {
                d_size = 2;
            }
            else
            {
                d_size = STANDARD_DOM + 5;
            }
            for ( int j = 0; j < d_size; j++ )
                _temp_status [ i ][ j ] = shared_ptr [ idx++ ];
            _status [ i ] = _temp_status[ i ];
        }
    }
}//move_status_from_shared

int main () {
    
    init  ();

    uint * shared_ptr = new uint [ 13 * _scope_size];
    move_status_to_shared ( shared_ptr, MIXED_DOM );
    int idx = 0;
    for ( int i = 0; i < _scope_size; i++ )
    {
        int idx = 13 * i;
        for ( int j = 0; j < 13; j++ )
        {
            cout << shared_ptr [ idx+j ] << " ";
            if ( j < 5 )
                shared_ptr [ idx+j ] *= 100;
        }
        cout << endl;
    }

    
    for ( int i = 0; i < _scope_size; i++ )
    {
        for ( int j = 0; j < 13; j++ )
        {
            //cout << _status[ i ][ j ] << " ";
            cout << _temp_status [ i ][ j ] << " ";
        }
        cout << endl;
    }

    move_status_to_shared ( shared_ptr, MIXED_DOM );

    for ( int i = 0; i < _scope_size; i++ )
    {
        for ( int j = 0; j < 13; j++ )
        {
            //cout << _status[ i ][ j ] << " ";
            cout << _temp_status [ i ][ j ] << " ";
        }
        cout << endl;
    }
    
    
    delete [] shared_ptr;
    clean ();
    return 0;
    
    if ( all_ground () )
    {
        cout << "All ground\n";
    }
    else
    {
        cout << "Not all ground\n";
    }

    if ( only_one_not_ground () )
    {
        cout << "Only one not ground\n";
    }
    else
    {
        cout << "More than one not ground\n";
    }

    int var = get_not_ground ();
    cout << "Not ground " << var << endl;

    print ();

    // Subtract test
    cout << endl;
    cout << "Subtract Test:\n";
    vector < int > vals = { -1, 0, 1, 2, 3, 4, 5, 20, 1, 2, 21, 30, 32, 33, 34 };
    for ( int i = 0; i < vals.size(); i++ )
    {
        subtract ( 0, vals[i] );
    }
    print ( 0 );

    for ( int i = 0; i < 30; i++ )
    {
        subtract ( 0, i );
    }
    print ( 0 );

    for ( int i = 0; i < 20; i++ )
    {
        subtract ( 1, i );
    }
    print ( 1 );

    cout << "V_0: " << get_min ( 0 ) << endl;
    cout << "V_0: " << get_max ( 0 ) << endl;

    cout << "V_1: " << get_min ( 1 ) << endl;
    cout << "V_1: " << get_max ( 1 ) << endl;

    if ( is_singleton ( 1 ) )
    {
        cout << "V_1 is singleton\n";
    }
    else
    {
        cout << "V_1 not singleton\n";
    }

    cout << get_sum_ground () << endl;

    if ( is_empty_var ( 0 ) )
    {
        cout << "V_0 is empty" << endl; 
    }
    else
    {
        cout << "V_0 is not empty" << endl;
    }

    shrink ( 2, 0, 21 );
    print ( 2 );

    shrink ( 2, 1, 20 );
    print ( 2 );

    shrink ( 2, 2, 19 );
    print ( 2 );

    shrink ( 2, 2, 18 );
    print ( 2 );

    shrink ( 2, 3, 18 );
    print ( 2 );

    shrink ( 2, 4, 2 );
    print ( 2 );

    shrink ( 3, 10, 10 );
    print ( 3 );
    
    clean ();
    return 0;
}


