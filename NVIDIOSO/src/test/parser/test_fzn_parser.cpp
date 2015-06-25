#include <iostream>
#include "parser.h"
#include "tokenization_test.h"
#include "fzn_tokenization.h"
#include "fzn_parser.h"

using namespace std;

int main ( int argc, char * argv[] ) {

	Parser * parser = new FZNParser ( "FlatZincTest.fzn" );
	parser->parse ();
	
    parser->print();
    while ( parser->more_tokens() )
    {
    	(parser->get_next_content ())->print();
    }
    
    delete parser;
    
    return 0;
}
