**iNVIDIOSO**
===================
NVIDIa-based cOnstraint SOlver v. 1.0 
===============================

System requirements
-------------
iNVIDIOSO takes advantage of the GPU architecture to perform parallel computation. However, it is possible to install a sequential version of the solver by choosing the appropriate option during installation.

- C++11
iNVIDIOSO is written in C++11 and CUDA. Therefore, in order to compile the source code of the solver the system must be equipped with a C++11 compiler[^compiler].
[^compiler]: NVIDIOSO has been tested on gcc47, so I recommend to use gcc47 or higher.

To check if the compiler supports C++11, open the termina and type what follows:
```
$ g++ --version
```
C++ is (partially) supported from version 4.3 of *gcc*. The solver has been tested with version 4.7. You may want to update your current compiler version as follows:
```
$ sudo apt-get update
$ sudo apt-get install gcc-4.7
$ sudo apt-get install g++-4.7
```
For Mac users, the C++ compiler can be installed together with the XCode package. XCode is available from the Mac OS install DVD, or from the Apple Mac Dev Center. Another option for installing *clang* (the default compiler on MacOS systems) and the C++11 compiler on MacOS X is to install the Xcode Command Line Tools:
```
$ xcode-select –install
```

Finally, g++ can be also installed using *home-brew*.

- CUDA
If a GPU is present on the system, in order to install and use the GPU version of iNVIDIOSO1.0, the solver must be compiled using **nvcc** with **CUDA-6.5**.
I recommend to download and install the latest version of CUDA. 
To check if you have nvcc correctly installed with its path set on the current environment, open a terminal and type what follows:
```
$ nvcc --version
```
If you see something like the following:
> nvcc : NVIDIA (R) Cuda compiler driver Copyright ( c ) 2005−2014 NVIDIA Corporation Built on Thu Jul 17 19 :13:24 CDT 2014 
> Cuda compilation tools , release 6.5 , V6.5.12 

then nvcc is correctly installed and set and ready to use.
If you receive an error message, then you probably need to download and install nvcc from:
> http://developer.nvidia.com/cuda-downloads 

Please, remember to set also the path to the nvcc compiler in the iNVIDIOSO.envs file:
> iNVIDIOSO/NVIDIOSO/iNVIDIOSO.envs

Obtaining iNVIDIOSO1.0
-------------
iNVIDOSO can be downloaded from the following *github* repository by clicking on the *download ZIP* button:
> github.com/FedericoCampe8/NVIDIOSO

 If you want to contribute to the project, you should have git installed in your system.
To check if you have git on your system, type the following in your terminal:
```
$ git --version
```
If you receive an error message, then you can choose between two options: 
>- create a github account on github.com and then download a zip version of the solver from github.com/FedericoCampe8/NVIDIOSO;
>- install git on your system and use a cloned version of the solver to collaborate to its development.

To install git in your system, type the following command in your terminal:
```
$ sudo apt-get update
$ sudo apt-get install git
```
> **Note:**
>- Developers should collaborate to the project using git and pushing their extensions on github server. 

Once you have git properly installed in your system you can clone the project, modify it, and then commit the new changes on your local version and/or push them on the server. For new developers I highly recommend to create a github account and **fork** the project by clicking the *fork* button on and then clone the repository.

In the next section I show a simple working example about how to use git to modify and update files on the master branch (i.e., on the server). However, I strongly encourage to read the git tutorial present on git webpage

>http:://git-scm.com/documentation.

--------

- Git working example
Suppose that two developers, John and Jessica, want to work together with a shared repository that is on some server. John starts by cloning the repository on his local machine, modifying some files, and committing the changes:
```
# John's machine
$ git clone john@githost:simplegit.git
$ cd simplegit/
# Some modifications to the code here
$ git commit -am 'fixed bug abc'
```

Jessica does the same with her local copy:

```
# Jessica's machine
$ git clone jessica@githost:simplegit.git
$ cd simplegit/
# Some modifications to the code here
$ git commit -am 'add some description'
```

Now Jessica pushes her work up to the server
```
# Jessica's machine
$ git push origin master
```

In the mean time, John tries to do the sam
```
# John's machine
$ git push origin master
```
But he receives an error message since the server has a most updated copy of the project (updates coming from Jessica). Therefore, John has to first fetch the new updates, and then merge them on his local copy:
```
# John's machine
$ git fetch origin
$ git merge origin/master
```

Then he can update his version on the server:
```
# John's machine
$ git push origin master
```

In the mean time, Jessica has done some other local work she wants to update on the server:
```
# Jessica's machine
# First she creates a new branch from the master branch
$ git branch fixBug
# Then she works on the code
# When done, she merges the work with her current master branch
$ git fetch origin
$ git checkout master
$ git merge fixBug
# Now she updates the code on the server
$ git merge origin/master
$ git push origin/master
```


The above example presents a normal working flow using git. In particular, I recommend to create a new branch for every new update/modification on the local master branch:

```
$ git branch hotFix
```

After working on that branch, it should be merged with the master and deleted:

```
$ git checkout master
$ git merge hotFix
$ git branch -d hotFix
```

To see the status of your git repository (including branches) use
```
$ git status
```

To clone the repository via SSH (recommended) use the following address:
> git@github.com:FedericoCampe8/NVIDIOSO.git.

To generate a SSH public key follow the instructions present in
> http://git-scm.com/book/it/v2/Git-on-the-Server-Generating-Your-SSH-Public-Key.

Then, email your public key to fede.campe@gmail.com to be added as a developer.

> **Note:**
> - Fork the project on your personal github account would be a better idea instead of cloning it directly from the server;
> - Branch from master every time you want to modify the code with important changes;
> - Remember to fetch the updates from the server before pushing them on github;
> - To remove a file/folder use *git rm < file>*.

Installing iNVIDIOSO1.0
-------------
 
 To install iNVIDIOSO1.0,  first set the environment variables 
in the file
	> NVIDIOSO/config/iNVIDIOSO.envs
	
In this file you can set several environment variables such as the path where to install iNVIDIOSO1.0, the compiler version, and the compute capability of the graphic card if present.

After the environment variables have been set, you just need to go into the iNVIDIOSO main folder run the installation script as follows:
```
$ cd iNVIDIOSO/NVIDIOSO
$ ./install.sh
```
During the installation process you will be asked whether you prefer to install the CPU version or the GPU version of the solver:
just type "C" for the CPU version or "G" for the GPU version and press enter.

Using iNVIDIOSO1.0
-------------

The solver can be run using the following command:
```
$ ./invidioso -i input_file [options]
```

To see the list of options available to the solver, use the *-h* option:
```
$ ./invidioso -h
```

Structure of iNVIDIOSO1.0
-------------

iNVIDIOSO1.0 is based on two main components:
>- *Data Store*: it holds all the information about variables and constraints;
>- *(CP) Solver*: it uses the data retrieved from the data store to run its internal search strategies to find solutions.

- Data Store
A Data Store is constructed from a model written as a text file and passed to its constructor. The internal parser reads the file and initializes all the data structures and objects needed during the search phase and demanded by the (CP) Solver. Let us observe that iNVIDIOSO does not parse only FlatZinc files but it can read any constraint model (e.g., GeCode models) by deriving the parser class according to the different modeling language. For example, the ***cp_store*** class deriving from ***DataStore*** (see ***include/base***) contains a pointer to an instance of the parser to use for reading the model from the file. This class contains a member function called "init model" which reads the file and uses the parser to tokenize the strings and generate the corresponding objects in terms of variables, constraint, etc.
Another important element of the Data Store is the ***CPModel*** (see ***include/core***). The CPModel is a class representing a general CP Model which includes vari- ables, constraints, a search engine and a constraint store. These information is provided by the cp store during initialization. The CPModel also attaches the constraint store to all the variables in the model following an Observer/Subject pattern: every time the domain of a variable is modified, a notification is sent automatically to the constraint store.

- (CP) Solver
The CP Solver (see ***include/base***) is a class that implements a CP solver deriving from Solver. The Solver class is an interface for defining solvers. It provides methods such as *add_model* or *run* to add models and running them, respectively. Therefore, the whole solver is not bounded to a specific type of solver but it can run different solvers (e.g., a Planner), provided the right instance of a class derived from Solver which implements its interface.
The CP Solver class is a class derived from Solver for solving CP instances. In particular, this class runs a CP model by first creating a constraint graph from the model and then invoking the labeling method of the search engine associated with the CP model (see ***src/base/cp solver.cpp***). Note that this class can run different CP models at "the same time3", meaning that is possible to upload different CP instances and solve them sequentially.

- Core components
iNVIDIOSO1.0 is made by several core component (i.e., classes). These classes are declared in the **include/** folder and organized by functionality. The most important classes are:
>- *constraint_store*: interface for a constraint store. Probably, the most important method of this class is the *consistency* method which propagates constraints and returns *True* if all domains are consistent w.r.t. the constraints or *False* otherwise.
>- *variable*: abstract class representing the variables of the solver. Variables have a list of constraints (constraints for which they are involved in) and a pointer to the constraint store to notify whenever their domain is changed. Variables have a *DomainIterator* member which represents an iterator to the associated domain and which is used to perform operations on it.
>- *Domain*: abstract class representing a domain. This class does not specify how a domain should be implemented. Instead, it specifies a set of methods to define in each class deriving from it. For example, see *int_domain*.
>- *memento*: this is a class implementing a *memento* pattern for "Backtrackable" objects. A class which implements a *BacktrackableObject* (see *int_variable*) can store its state on a stack and retrieve the same state later during the computation. This is how a backtrack search strategy is implemented.

- Constraints
Constraints are declared in ***include/constraint***. A constraint is an object which is bounded to a set of variables and which perform some actions on these variables, changing their internal state. Two important methods are *consistency* and *satisfied*, which propagates the constraint and check for its satisfiability respectively. In iNVIDIOSO1.0 the constraints are those defined in the FlatZinc specification. In particular, the class *fzn_constraint* derives from Constraint an defines all the FlatZinc constraints. In turn, each FlatZinc constraint derives from *fzn_constraint* and implements the specific methods declared in the interface. For example, see ***include/constraints/int_ne.h*** and its definition in ***src/constraints/int ne.cpp***.

- Search
Search engines are declared in ***include/search***. A search engine is an inter- face which represent a general search engine. It defines several methods which are used by the solver to perform search. For example, SearchEngine defines the labeling method which perform the actual search. First it sets up the inter- nal attributes of the search. Then, it calls the labeling function with argument specifying the index of a non grounded variable.
The class *DepthFirstSearch* is an example of a search strategy implemented as a derived class from SearchEngine. From the implementation point of view, different heuristics can be defined by deriving from the *Heuristic* class. This class is an interface used by the search engine classes in order to select the next variable to label (*get_choice* variable method) and the value to assign to it (get *choice_value* method).

- Other Components
Other main components of the system are:
>- *FZ_parse*: parse for FlatZinc models. It describes how to tokenize FlatZinc files and which tokens should be produced by this tokenization.
>- *exception*: it declares the exceptions that can be raised during the computation. This is an important class and developers must use exceptions and catch them to properly write the code and later debug it.
>- *logger*: allows one to print and log messages during computation

How to understand iNVIDIOSO1.0
-------------

The above sections briefly describe the core components of the system. The developer should refer to the html API page where it is possible to file the complete list of classes and their hierarchy. Moreover, there is a description for every class and every method of that class. These descriptions are present in the Doxygen folder.

- Where do I start?
I personally like a top-down approach when I’m trying to understand the code someone else has written. So, I will briefly describe the code starting from the main function, i.e., from the file *main.cpp*. If you open *main.cpp* you will see the main function which does six "main" things (don’t consider logger and statistics messages for now):
>- It creates a singleton object *InputData*: this is nothing more than a parser for the input given by the user;
>- It creates a singleton object *d_store* of type *DataStore*: this object represents the "store" containing all the information about the model(s) to solve (i.e., variables, constraints, and search strategies);
>- It loads the model(s) into the store:
```
d_store.load_model ();
	
```
>- It initializes the model(s) by filling the data structures within the DataStore object:

```
d_store.init_model ();
```

>- It generates a new solver of type *CPSolver* for solving the model previously initialized:
```
CPSolver * cp_solver = new CPSolver ( d_store.get_model () );
```
>- It solves the model(s):
```
cp_solver->run ();
```
In what follows I will give a brief description of almost all of the above steps.
I will skip the InputData class description since it is a straightforward implementation of a parser.

- Model loading and initialization
Model loading and initialization are done by methods declared in the CPStore class. CPStore is a subclass of DataStore which specializes the DataStore for *Constraint Programming* models. 
The definition of the methods *load_model()* and init_model() can be found in *cp_store.cpp*.
The *load_model* function creates a parser for *FlatZinc* style input files (a different parser can be used as well) and iterates through the input file until the parser can find tokens. These tokens are stored in the parser state and are ready to be given to the store.
These token will be converted into "high-level" objects used during the solving process.
The *init_model* method creates a new CPModel object and fills it with vari- ables, constraints, the constraint store, and a search engine, according to the parsed model. This is done by creating a *generator* object which task is to con- vert tokens given by the parser to solver’s object of the proper type (just look at it as a general abstract factory class). Note that a CPModel object represents the a model to be solved by the solver. A CPStore, instead, represents a "global" container of information for one or more (CP) models to solve.

- Solving a model
This subsection is a draft. Briefly, the "important" methods are defined in *cp_solver.cpp*. In particular, the *run()* method runs all the models added to the CPSolver instance executing the following steps:
>- It creates the *constraint graph* (i.e., it attaches constraints to variables);
>- It initializes the constraint store for an initial propagation (before starting the search process);
>- It attaches the constraint store to each variable so any variable can notify the constraint store upon any change on its domain;
>- It invokes the labeling function of the search engine which *labels* the variables and performs backtrack according to a given search strategy.

The *run()* method calls the run model method on each model (i.e., models saved in the vector of CPModels "_models"). Let's consider a specific model, e.g., the CPModel *_models[0]* (note that the vector *_model* contains pointer to the CPModel objects). The *run_model* method perform the following main steps:
>- It creates the constraint graph for the model by attaching each constraint to the variables in its scope (*create_constraint_graph()*):
```
// See cp_model.cpp
for ( auto c : _constraints )
	c->attach_me_to_vars ();
```
This allows a variable to notify the constraints it is involved in whenever its domain changes (see Observer pattern).
>- It initializes the constraint store (init constraint store). This fills the constraint store with all the constraints in order to perform an initial propaga- tion before the search phase starts. The initial propagation of constraints is needed in order to prune domains or to find unsatisfiable models as soon as possible (e.g., X = 2, Y = 2, X != Y);
>- It attaches the constraint store to each variable in the model (*attach_constraint_store()*):
```
// See cp_model.cpp
for ( auto var : _variables )
	var->attach_store ( _store );
```
This is done for two main reasons: first, it decouples the variables to the constraint store, meaning that we can change the constraint store during the search without compromising the status of the variables. Second, a variable can notify the constraint store whenever its domain changes (see Observer pattern). In particular, this allows the variables to add the constraints they are involved in into the queue of constraints of the constraint store for the following propagation phase.
>- It starts the search by calling the method "labeling" defined for each (valid) search engine:
```
// See cp_model.cpp
( model-> get_search_engine () )->labeling ();
```
The first 3 steps are pretty straightforward, so we shall focus on the last point: the labeling phase.

- Labeling
The labeling method allows the solver to perform the actual search. The search phase is carried on by a SearchEngine object (see *search_engine.h*) which must implements all the pure methods of the abstract class SearchEngine. A search engine is composed by several objects:
>- A *constraint store* to invoke for propagating constraints during search;
>- A *heuristic* that defines the order of variables to label and the values to assign to them;
>- A *solution manager* that keeps track of all solutions found during the search and other statistics;
>- A *backtrack manager* that allows the search to *backtrack* to previous states for backtrack strategies (e.g., depth first search). Backtrack manager is implemented using the *memento* pattern.

To check how a search engine really works, let’s consider the depth first search engine (see *depth_first_search.cpp*) and, in particular, the *label* method, invoked by *labeling()*. The label method takes the current level of the search tree as input parameter (var_idx) and calls itself recursively until either a(ll) solution(s) is(are) found or the model is proven to be unsatisfiable. Let’s us observe that the index given to the label method is not the index of a variable among the array of variables to label. Instead, it represents the level of the search tree and only as a "side effect" it corresponds to a specific variable.
The first thing that the label method does is to propagate the constraints by invoking the consistency method of the constraint store:
```
consistent = _store->consistency ();
```
If the store is not consistent (e.g., a variable has an empty domain), then the search returns *false* and, in case, it backtracks. 
Otherwise, the propagation was successful and the search can continue by labeling the next variable. In particular, the new (pointer to a) variable to label is chosen by the heuristic:
```
var = _heuristic->get_choice_value ();
```
and the labeling is performed by shrinking the domain of the selected variable to the chosen value:
```
(static_cast<IntVariable*>(var))->shrink (value, value);
```
The search then continues with a recursive call:
```
int next_index = _heuristic->get_index();
...
consistent = label ( next_index );
```

If the result of a recursive call is *true*, all the children of the current node of the search tree are explored (and consistent) and the search returns. If, instead, the result of a recursive call is *false*, the current assignment of value to var leads to a failure somewhere in the subtree. This value must be removed from the domain of the current labeled variable and another labeling must be performed:

```
(static_cast<IntVariable*>(var))->subtract (value);
...
consistent = label ( var_idx );
```
Note that before doing the next labeling the status of the search phase must be restored, i.e., a backtrack must be performed:
```
_backtrack_manager->remove_level (_depth);
```
Whenever a new labeling is not possible the search returns *false*.

- Backtracking
Let’s now see how backtracking works during search (of course,
the following is valid for search engines based on bactracking strategies). Backtracking is performed by an object of type BacktrackManager (see *backtrack_manager.h*).
Using the "Memento" pattern terminology, a BacktrackManager is the *Caretaker* object who is in charge of managing the list of Memento object. In particular, it represents the interface for the client code to access and work on memento objects. Furthermore, it stores Memento objects and re-set a previous states, effectively performing backtrack. In turn, a *memento* object is defined by a *BacktrackableObject* (see *backtrackable_object.h*), i.e., a *wrapper* class that stores the state of any object that inherits from it. Let us observe that a BacktrackableObject contains a Memento object which represents the actual information to store. 

- Propagating constraints
Propagation of constraints is perfomed inside a constraint store (see *constraint_store.h* and *simple_constraint_store.h*) by invoking the "consistency" method. A constraint store is best described as a data structure containing all the constraints to propagate in a queue of constraints. In particular, the constraint store iterates over this queue of constraints, poping one constraint at a time and propagating it, until it reaches a fix point, i.e., the queue is empty. Let us observe that, in general, the propagation of a constraint modifies the domains of the variables involved in it, triggering, as a conseguence, the propagation of other constraints. This process stops when either no domains can be further modified or a domain is found to be empty due to previous prop- agations. From the practical point of view, this process is represented by the "while" loop in the consistency method (see *simple_constraint_store.cpp*).

- Constraints
Constraints are "relations" on (subsets of) variables’ domains. Propagation of constraints is needed to "prune" the search tree and to find "failure" paths (i.e., paths that lead to assignments that do not satisfy all the constraints of the model). All the constraints must inherit from the *Constraint* class (see *constraint.h*) and they must implement (among others) two important methods:
>- *bool consistency ()*, that represents the consistency function which removes the values from variable domains, actually propagating the constraint.
>- *bool satisfied ()*, that checks if the constraint is satisfied.

In the current implementation of the solver (iNVIDIOSO v1.0), we consider all the constraints defined in the FlatZinc specification[^flatzinc].
[^flatzinc]: See http://www.minizinc.org/specifications.html.

- Variables and Domains
A variable is an object which inherits from the *Variable* class (see *variable.h*). Every variable has a set of related constraints, a constraint store to notify whenever the corresponding domain is changed, and a *DomainIterator* object which can be seen as a pointer to the domain of the variable that can be used to perform some actions on it (e.g., find the minimum element, shrink the domain, etc.). In turn, a domain can be defined on Boolean, Integer, or Set values. In the current implementation (iNVIDIOSO v1.0) we considered only Integer values. In particular, we designed the representation of a domain keeping in mind the restrictions imposed by the GPU architecture, e.g., small amount of (shared) memory available. For this reason, we use two different representations for a given domain D, according with its size and a given max value *k* (256 by default):
>- If |D| < k + 1, then D is represented by a bitmap of *k* bits;
>- If |D| > k, then D is represented by a list of domain bounds pairs (*min*, *max*).

Moreover, we add 5 other integer values to each domain, in order to describe its properties. Therefore, domains have the following structure:
> | EVT | REP | LB | UB | DSZ || ... BIT ... |

where:
 >- EVT: represents the EVenT happened on the domain. Different events may trigger different constraint propagators;
 >- REP: is the REPresentation currently in use;
 >- LB: Lower Bound;
 >- UB: Upper Bound;
 >- DSZ: Domain SiZe;
 >- BIT: bitmap vector (i.e., list of unsigned integers storing bits).

Let us observe that in the current version of the solver, we use *CudaDomain* object to represent variables’s domains (see *cuda_domain.h*):
> Domain <- IntDomain <- CudaDomain
In turn, a CudaDomain contains a (pointer to a) *CudaConcreteDomain* object which is the object containing the actual array of domain’s elements:
```
int * _concrete_domain;
```

This is a quite complex framework (i.e., subclasses, iterators on domains, etc.) and I won’t go into further implementation details. However, I would like to stress the fact that the best way to access domains and modify them is through a DomainIterator object (which is present in each variable object as a public member). To recap, a domain must implement the pure methods of the base class Domain, and the user can access to it by a DomainIterator object attach to such domain. This is all what a domain’s user needs to know.
For a detailed explanation, please contact 
> fede.campe@gmail.com.

- Note for developers
I report some useful notes for developers in what follows:
>- Write the code as clear as possible;
>- Use "_variable" for private variables where "variable" is the name of the private variable;
>- Comment every method when you define a new class as well as a brief description of the class at the top of the file;
>- Comments should be parsed by Oxygen, therefore use
```
//! For single line commentd
/** 
 * For comments spanning 
 * on multiple 
 **/
 @param to describe the list of parameters of a function
 @return to describe the values returned from a function
```
>- Use design patterns. Dot. There is a design patter for almost everything you would like to implement. Using them allows other developers to better understand your code and your intentions in what you wanted wrote;
>- The following three are particularly important:
>>- Prefer decoupling of classes;
>>- Program to an interface, not and Implementation;
>>- Favor object composition over class inheritance.

>- Use C++11 and no old style C programming. For example:
>>- Use *new* and *delete* instead of *malloc* and *free* functions;
>>- Use C++ *string* class instead of char * string;
>>- Prefer *unique_ptr* to *shared_ptr* and avoid to use standard pointers;
>>- Prefer *vector*(*s*) instead of standard array(s);
>>- Use the C++(11) Standard Template Library (STL) instead of reinventing the wheel. STL has built in data structure and algorithms that can satisfy the majority of the needs (e.g., use (*unordered_*)*map* for hashing);
>- Comment your code,
>- Find examples or have a look to the C++ guide, e.g., 
> > http://en.cppreference.com 
>  http://www.cplusplus.com/.

Thanks for reading this page. If you’ have any further questions, please don’t hesitate to contact fede.campe@gmail.com.