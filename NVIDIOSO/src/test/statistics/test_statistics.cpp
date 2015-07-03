#include <iostream>

#include "globals.h"
#include "statistics.h"

using namespace std;

int fibonacci ( int n )
{
    if ( n < 2 ) return 1;
    return fibonacci ( n - 1 ) + fibonacci ( n - 2 ); 
}

int main () {
    statistics.set_timer ( Statistics::TIMING::ALL );
    statistics.set_timer ( Statistics::TIMING::PREPROCESS );

    int n   = 40;
    int fib = fibonacci ( n );
    cout << "Fibonacci of " << n+1 << " is " << fib << endl;
    
    statistics.stopwatch ( Statistics::TIMING::PREPROCESS );
    statistics.stopwatch ( Statistics::TIMING::ALL );

    statistics.print();
    
    return 0;
}
