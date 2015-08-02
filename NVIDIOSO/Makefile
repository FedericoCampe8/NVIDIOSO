# Include a definition file
include ./make.inc
include ./make_cuda.inc

# ---------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------
MAIN=domain_test
MAIN=int_ne_test
MAIN=main
 
OBJECTS =  $(patsubst %.cpp,%.o, $(PRG_PATH)/$(MAIN).cpp)
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(BASE)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(CONSTRAINTS)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(GLOBAL_CONSTRAINTS)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(CUDA_CONSTRAINTS)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(CUDA_GLOBAL_CONSTRAINTS)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(VARIABLE)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(CUDA_VARIABLE)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(CORE)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(CONSTRAINT_STORE)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(CUDA_CONSTRAINT_STORE)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(CUDA_UTILITIES)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(PARSER)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(FZN_PARSER)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(SEARCH)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(EXCEPTION)/*.cpp))

HEADERS = $(patsubst %.cpp,%.h,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(BASE)/*.cpp))
HEADERS += $(patsubst %.cpp,%.h,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(CONSTRAINTS)/*.cpp))
HEADERS += $(patsubst %.cpp,%.h,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(GLOBAL_CONSTRAINTS)/*.cpp))
HEADERS += $(patsubst %.cpp,%.h,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(CUDA_CONSTRAINTS)/*.cpp))
HEADERS += $(patsubst %.cpp,%.h,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(CUDA_GLOBAL_CONSTRAINTS)/*.cpp))
HEADERS += $(patsubst %.cpp,%.h,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(CONSTRAINT_STORE)/*.cpp))
HEADERS += $(patsubst %.cpp,%.h,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(CUDA_CONSTRAINT_STORE)/*.cpp))
HEADERS += $(patsubst %.cpp,%.h,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(VARIABLE)/*.cpp))
HEADERS += $(patsubst %.cpp,%.h,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(CUDA_VARIABLE)/*.cpp))
HEADERS += $(patsubst %.cpp,%.h,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(CUDA_UTILITIES)/*.cpp))
HEADERS += $(patsubst %.cpp,%.h,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(CORE)/*.cpp))
HEADERS += $(patsubst %.cpp,%.h,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(PARSER)/*.cpp))
HEADERS += $(patsubst %.cpp,%.h,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(FZN_PARSER)/*.cpp))
HEADERS += $(patsubst %.cpp,%.h,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(SEARCH)/*.cpp))
HEADERS += $(patsubst %.cpp,%.h,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(EXCEPTION)/*.cpp))

# ---------------------------------------------------------------------
# Compiler options 
# ---------------------------------------------------------------------
vpath %.o obj

## lib dirs -L...
CCLNDIRS= 

## include dirs -I...
INCLDIRS =  -I$(PRG_PATH) 
INCLDIRS += -I/usr/include/
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC) 
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(BASE)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(VARIABLE)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(CUDA_VARIABLE)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(CONSTRAINTS)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(GLOBAL_CONSTRAINTS)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(CUDA_CONSTRAINTS)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(CUDA_GLOBAL_CONSTRAINTS)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(CONSTRAINT_STORE)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(CUDA_CONSTRAINT_STORE)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(CORE)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(PARSER)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(FZN_PARSER)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(CUDA_UTILITIES)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(SEARCH)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(EXCEPTION)

## Compiler Flags
CCFLAGS = $(CCOPT) $(INCLDIRS)

DEPS = $(HEADERS)

## Note for developers: add the following macro
## -DCUDAVERBOSE=true
## to activate CUDA verbose (e.g., prints from kernel)

# Incrementally build the system.
all : 	nvidioso
	$(PRINT_DONE)

# Default
nvidioso :	$(OBJECTS) 
	$(CC) $(CUDAOPT) $(CCLNDIRS) $(CFLAGS) -o $(TARGET) \
	$(OBJECTS:%=$(PRG_PATH)/obj/%)

# Debug
debug : $(OBJECTS) 
	$(CC) $(CUDAOPT) $(DEBUGFLAG) $(CCLNDIRS) $(CFLAGS) -o $(TARGET) \
	$(OBJECTS:%=$(PRG_PATH)/obj/%)

# Library
lib : $(OBJECTS)
	$(AR) $(ARFLAGS) lib/$(TARGET).a \
	$(OBJECTS:%=$(PRG_PATH)/obj/%)

# $<: the first prerequisite filename.
# $@: the target filename.
$(OBJECTS) : 	%.o : %.cpp
	$(CC) $(CUDACC) $(CUDAOPT) $(COMPILE) $(CFLAGS) $(CCFLAGS) $< -o $(PRG_PATH)/obj/$@
	$(PRINT_COMPILE)

#.PHONY : clean
clean:
	@rm -f obj/*.o 	
	@rm -f obj/*/*/*.o	
	@rm -f $(TARGET)	
	$(PRINT_CLEAN)	
