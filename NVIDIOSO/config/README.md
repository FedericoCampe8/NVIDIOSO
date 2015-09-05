**iNVIDIOSO v1.0**
===================
iNVIDIOSO Configuration 
===============================

**iNVIDIOSO1.0** can be configured with many parameters.
Each parameter has its proper options.
In what follows we report the options that can be set before running iNVIDIOSO1.0.

| Class             | Name | Option | Description |
:-------    | :------- | :--- | :------- |
| iNVIDIOSO SEARCH  |      |        |             |
|| SEARCH_TREE_DEBUG  | YES/NO | Enable debug printing during search. |
|| SEARCH_TRAIL_DEBUG    | YES/NO | Enable debug printing for trailstack during search. |
|| SEARCH_TIME_WATCHER    | YES/NO | Stores more information about time spent during search. |
|| SEARCH_TIMEOUT    | {-1,0,1,…} | Search timeout in sec. (-1 no timeout). |
|| SEARCH_SOLUTION_LIMIT    | {-1,0,1,…} | Max number of solution (-1 all solution). |
|| SEARCH_BACKTRACK_OUT    | {0,1,2,…} | Max number of backtracks before terminating search (0 no limit). |
|| SEARCH_NODES_OUT    | {0,1,2,…} | Max number of explored nodes before terminating search (0 no limit) |
|| SEARCH_WRONG_DECISIONS    | {0,1,2,…} | Max number of wrong decisions before terminating search (0 no limit). |
| iNVIDIOSO CONSTRAINTS  |      |        |        |
|| CONSTRAINT_PROPAGATOR_CLASS  | naive, bound, domain | Type of constraint propagation to perform. Not all propagators are available for all constraints.|
| iNVIDIOSO CONSTRAIN STORE  |      |        |        |
|| CSTORE_CONSISTENCY  | YES/NO | Enable constraint propagation. |
|| CSTORE_SATISFIABILITY  | YES/NO | Enable satisfiability check. |
|| CSTORE_CUDA_PROP_LOOP_OUT  | {1,2,3,…} | Iterations of propagation to perform on device after labeling. |
|| CSTORE_CUDA_PROP  | sequential,block_per_constraint, block_per_variable, block_per_k_constraint, block_per_k_variable | Type of propagation on device: sequential propagation,one block per constraint, one block per variable,one block per k constraints (fill block),one block per k variables (fill block). |
| iNVIDIOSO LOCAL SEARCH  |      |        |        |
|| LS_SAT_CONSTRAINTS  | all_soft, all_hard, mixed | Consider all constraints as soft, hard, or mixed (require specification in the model for mixed type).|
|| LS_RESTARTS  | {0,1,2,...} | Number of restarts to perform (re-start search from initial solution).|
|| LS_ITERATIVE_IMPROVING  | {0,1,2,...} | Number of iterative improving steps (re-start search from current best solution solution).|
| iNVIDIOSO CUDA PARAMETERS  |      |        |        |
|| CUDA_SHARED_MEM  | {16,47,…} | Maximum size (in KB) of shared memory available on device.|
|| CUDA_MAX_BLOCK_SIZE  | {32,64,128,…} | Maximum number of threads per block available on device.|
