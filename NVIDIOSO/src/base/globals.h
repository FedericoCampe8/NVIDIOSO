//
//  globals.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 26/06/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#ifndef NVIDIOSO_globals_h
#define NVIDIOSO_globals_h

/* Common dependencies */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>
#include <unistd.h>

/* Input/Output */
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

/* Arithmetic and Assertions */
#include <cassert>
#include <cmath>
#include <limits>

/* STL dependencies */
#include <algorithm>
#include <iterator>
#include <map>
#include <list>
#include <queue>
#include <stack>
#include <set>
#include <string>
#include <regex>
#include <vector>
#include <utility>

/* Global classes */
#include "logger.h"
#include "id_generator.h"

/* Environment variable */
constexpr int VECTOR_MAX = 256;

#endif
