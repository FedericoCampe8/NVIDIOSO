#include <iostream>
#include "parser.h"
#include "tokenization_test.h"
#include "fzn_tokenization.h"

using namespace std;

int main ( int argc, char * argv[] ) {

	UTokenPtr ptr;
    Tokenization * tkn = new FZNTokenization ();

    vector< string > flatzinc_model_test =
    {
        "array [1..8] of var 1..8: q;",
        "constraint int_lin_ne([1, -1], [q[3], q[4]], 1);",
        "constraint int_ne(q[1], q[2]);",
        "solve satisfy;",
        "solve  :: int_search(q, input_order, indomain_min, complete) satisfy;",
    };

    for ( int i = 0; i < flatzinc_model_test.size(); i++ )
    {
        auto line = flatzinc_model_test[i];
        (static_cast< FZNTokenization* > (tkn))->set_internal_state ( line );
        cout << "Parsed line:\t" << line << endl;
        ptr = tkn->get_token();
        if ( ptr != nullptr ) ptr->print ();
        else                                  cout << "Null ptr - line not valid\n";
        cout << endl;
    }
    
    delete tkn;
    
    return 0;
}
