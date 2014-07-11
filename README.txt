
                                      	NVIDIOSO
                          NVIDIa-based cOnstraint SOlver v. 1.0
 
                                __CSP/COP REPRESENTATION__
 
  VARIABLES
 
  Variable has variable types.
  - bool: true, false
  - int: -42, 0, 69
  - set of int: {}, {2, 3, 4}, 1..10
 
  We distinguish  between four different types of variables, namely:
  - FD  Variables: standard Finite Domain variables
  - SUP Variables: SUPport variable introduced to compute the objective
                   function. These variables have unbounded int domains.
  - OBJ Variables: OBJective variables. These variables store the objective
                   value as calculated by the objective function through
                   standard propagation. These variables have unbounded
                   int domains.
 
 
  DOMAINS:
 
  Domain representation may vary depending on the type of model that is 
  instantiated.
  In particular, for a CPU model the domains can be represented by lists
  of sets of domain value.
  For CUDA models domains are represented as follows.
  There are two internal representations for an finite domain D depending
  on whether |D| <= max_vector or not:
  - Bitmap: if |D| <= max_vector;
  - List of bounds: otherwise.
 
  By default, max_vector is equal to 256. This value can be redefined via
  and environment variable VECTOR_MAX.
 
  Domains have the following structure:
 
  | EVT | REP | LB | UB | DSZ || ... BIT ... |
 
  where
 
  - EVT: represents the EVenT happened on the domain;
  - REP: is the REPresentation currently used;
         This value can be one of the following:
         - -1, -2, -3, ...: BIT represents a set of 1, 2, 3, ... bitmaps
                            respectively. Each bitmap represents a 
                            domain subset of values {LB, UB};
         - 0              : BIT represents a Bitmap of contiguous values;
         - 1, 2, 3, ...   : in BIT there are respectively 0, 1, 2, ...
                            lists of bound. If 0 the bounds are set as
                            {LB, UB} in the LB/UB field respectively.
  - LB: Lower Bound of the current domain;
  - UB: Upper Bound of the current domain;
  - DSZ: Domain SiZe where DSZ <= max_vector -> REP = 0.
         Moreover, 
          - {LB, UB}' = {LB, k} {k', UB} -> DSZ' = DSZ - ( k' - k + 1 );
          - LB' = LB + k                 -> DSZ' = DSZ - ( k - LB + 1 );
          - UB' = UB - k                 -> DSZ' = DSZ - ( UB - k + 1 );
  - BIT: bit vector where
       - REP < 0: there is a total of (<=) VECTOR_MAX bits representing
                  REP pairs of bounds. The first part of BIT is used to
                  store REP triples <LB, Size, Offset> where Offset is a
                  pointer to the first bit of the bitmap representing the
                  pair {LB, LB + Size}. The second part of BIT stores the
                  actual bitmaps.
       - REP = 0: there are UB - LB + 1 <= VECTOR_MAX
                  bits of contiguous domain values;
       - REP > 0: each pair of bound is identified as
                  LB, UB (LB = UB if singlet).
 
  OBSERVATIONS (CUDA implementation):
 
  Shared Memory: 49152 = 48 kB per block -> keep 47 kB available.
  - REP < 0 there are 47 * 1024 = 48128 ->
    (48128 - 5 * 32)/32 = 1499 possible storable values.
    Worst case: REP = -256 -> 3 * 256 triples = 3 * 256 = 768 < 1499 (-8=256/32).
  - REP = 0 and VECTOR_MAX = 4096 the worst case is when there are 4096 sing.:
    ((4096 + 4096 * 2 * 32) / 8) / 1024 = 32.5 kB < 45 kB
    ((tot_bits + tot_bits * 2 int * bit_per_int) / B) / kB
  - REP > 0: 45 kB = 11520 int -> 11520 - 5 = 11515 -> 11515/2 (used two int
    to represent a pair of bounds) = 5757 pairs separated by at least
    one "hole" from each other -> 5757 * 2 = 11514 such as {0, 1}, {3, 4}, ... .
 
  It means that when the domains are greater than 11514 then a check must be
  performed in order to apply multiple copies from global to share memory if
  needed.
	
  See refman.pdf for further information about the implementation.
  
  Thanks for reading this page.
  If you've got any further questions, please don't hesitate to contact camp8@nmsu.edu.
	
