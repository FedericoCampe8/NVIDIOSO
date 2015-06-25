#include <iostream>
#include "parser.h"
#include "tokenization_test.h"

using namespace std;

int main ( int argc, char * argv[] ) {

    TokenizationTest tt ( "  #Hello " );
    tt.filter ();
    if ( tt.do_skip ( "# This is a comment " ) ) cout << "Skip \n";
    else  cout << "No skip \n";
    
    return 0;
}
