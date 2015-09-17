**iNVIDIOSO**
===================
iNVIDIOSO Unit Test Framework 
===============================
**iNVIDIOSO1.0** is has several classes (09/13/15 there are ~250k lines of code) and it is growing more and more every day.
Therefore, there is the need of a framework for unit testing on the components added to the solver.
This framework is present in 
	> NVIDIOSO/test
	
and it is run from the *main* function in 
	> invidioso_utest.cpp

To make things easier, there is script which runs the tests and report information regarding the passed/failed tests and possible memory leaks.
To run the test, *cd* into the main folder and invoke the *run_utest* script as follows:
```
$ cd iNVIDIOSO/NVIDIOSO
$ ./run_test.sh
```
 The above script will run all the unit tests written on iNVIDIOSO1.0 using Valgrind to detect possible memory leaks.
 At the end it will output something like the following:

```
============= iNVIDIOSO1.0 Unit Test =============
===================================================
Architecture: Darwin Federicos-iMac.local 14.5.0 Darwin Kernel Version 14.5.0: Wed Jul 29 02:26:53 PDT 2015; 
root:xnu-2782.40.9~1/RELEASE_X86_64 x86_64
===================================================
Running unit tests...
...
ANALYSIS RESULTS:
- UNIT TEST COMPLETED SUCCESSFULLY
- ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 21 from 21)
===================================================
===================================================
```
Indicating that unit test is completed without errors and there are 0 memory errors (in the above example).

> **Note:**

> - To run the unit test script it is necessary to install iNVIDIOSO1.0 in *unit_test* mode. This can be achieved by re-installing iNVIDIOSO using the *install.sh* script with the  keyword *unit_test*;
> - To run the unit test script it is necessary to install *Valgrind* if not already present. If you don't want to install Valgrind it is still possible to perform unit testing by running only the unit test tool with the following command: *./invidioso -v* (use *-h* to print the help message);
> - The unit test script will produce a log file containing all the details of the tests in order to investigate the source of error if any;
> - When iNVIDIOSO1.0 is build in *unit_test* mode, it will be slower compared with the standard installation. Use unit test to test new components and then re-install iNVIDIOSO as usual;
> - Every new component added to the system should be tested. The script will run all the tests written so far, plus the test for the new component. Please, note that this may require some time to be completed.

----------

Write my own Unit Test 
-------------

The following steps explain how to write a new unit test for a new component of iNVIDIOSO1.0:

1 - Extend *UnitTest* class 
: Unit tests classes inherit from **UnitTest**
: The name of a new class for the unit test *my_test* should be written in a (header) file named *my_test_utest* and the class name should be **MyTestUTest**

2 - Override the *test* method
:   There is at least one method to override which is the method invoked to perform unit test. This method has the following signature:
	> bool test(); // Returns true if the test is passed, false otherwise

:   There are some MACROS that can be used to write new unit tests. These MACROS are defined in *unit_test.h*. Some of them are:
	> TEST_TRUE ( bool b, std::string str ) // Test whether b is true, str name of the test
	> TEST_NULL ( void * ptr, std::string str ) // Test whether ptr if a NULL pointer
	> ...


3 - Register the test class
:   A class **MyTestUTest** declared in a file *my_test_utest* must be registered in order to be invoked. In order to register a class, open the file *unit_test_register.cpp*, define a new poster function and add it in the *fill_register* function as follows:
>```
UnitTest* p_my_test_utest ()
{
		return new MyTestUTest ();
}//p_my_test_utest
...
void
UnitTestRegister::fill_register()
{
		...
		add ( "my_test",  p_my_test_utest );
}//fill_register
```


4 - Include your test  

:   Include *my_test_utest.h* in *unit_test_inc.h*
 

5 - Rebuild the solution in *unit_test* mode 

:   Reinstall iNVIDIOSO using the *install.sh* script in *unit_test* mode
	> $ ./install.sh unit_test

:   Run unit test using the following script (or the tool itself):

	> $ ./run_utest.sh
	
> **Note:**

> - It is possibile to run a single unit test by either giving the name of the test to run as parameter to the script or by invoking the unit test tool with the *-t* option.


Currently implemented unit test classes:
>- input_data
>- cp_solver 

Thank you for reading this page, for any further question, please feel free to contact fede.campe@gmail.com. 
