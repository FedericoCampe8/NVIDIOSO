iNVIDIOSO1.0 can be configured with many parameters.
Each parameter has its proper options.
In what follows we report the options that can be set before running iNVIDIOSO1.0.

	CLASS			NAME				OPTION				DESCRIPTION
iNVIDIOSO SEARCH

				SEARCH_TREE_DEBUG		YES/NO				Enable debug printing during search			
				SEARCH_TRAIL_DEBUG		YES/NO				Enable debug printing for trailstack during search
				SEARCH_TIME_WATCHER		YES/NO				Stores more information about time spent during search
				SEARCH_TIMEOUT			{-1, 0, 1, …}			Search timeout in sec. (-1 no timeout)
				SEARCH_SOLUTION_LIMIT   	{-1, 0, 1, …}			Number of solution to find (-1 all solution)
				SEARCH_BACKTRACK_OUT    	{0, 1, 2, …}			Number of backtracks before quitting search (0 no limit)
				SEARCH_NODES_OUT		{0, 1, 2, …}			Number of explored nodes before quitting search (0 no limit)
				SEARCH_WRONG_DECISIONS		{0, 1, 2, …}			Number of wrong decisions before quitting search (0 no limit)

iNVIDIOSO CONSTRAINTS
				CONSTRAINT_PROPAGATOR_CLASS	{naive, bound, domain}		Type of constraint propagation to perform. 
												Not all propagators are available for all constraints.

iNVIDIOSO CONSTRAIN STORE
				CSTORE_CONSISTENCY		YES/NO				Perform constraint propagation
				CSTORE_SATISFIABILITY		YES/NO				Perform satisfiability check
				CSTORE_CUDA_PROP_LOOP_OUT	{1, 2, 3, …}			Iterations of propagation to perform on device after labeling
				CSTORE_CUDA_PROP		{sequential,			Type of propagation on device: sequential propagation, 
								 block_per_constraint,		one block per constraint, one block per variable,
								 block_per_variable,		one block per k constraints (fill a block),
								 block_per_k_constraint,	one block per k variables (fill a block)
								 block_per_k_variable }
iNVIDIOSO LOCAL SEARCH
				LS_SAT_CONSTRAINTS		{all_soft, all_hard, mixed}	Consider all constraints as soft, hard, or mixed
												(require specification in the model for mixed type)
CUDA PARAMETERS
				CUDA_SHARED_MEM			{16, 47, …}			Maximum size in KB of shared memory available on device
				CUDA_MAX_BLOCK_SIZE		{32, 64, 128, …}		Maximum number of threads per block available on device