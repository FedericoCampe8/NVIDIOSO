# ---------------------------------------------------------------------
# Makefile options 
# ---------------------------------------------------------------------

NO_COLOR=\x1b[0m
OK_COLOR=\x1b[32;01m
ERROR_COLOR=\x1b[31;01m
WARN_COLOR=\x1b[33;01m

ESC=`echo "\033"`
OK_COLOR ="${ESC}[0;32m"
NO_COLOR ="${ESC}[m\017"

OK_STRING=${OK_COLOR}[OK]${NO_COLOR}
OK_DONE =${OK_COLOR}NVIDIOSO compilation succeeded${NO_COLOR}

PRINT_COMPILE = @echo ${OK_STRING}
PRINT_COMPILE = @echo compile...${OK_STRING}
PRINT_DONE    = @echo ${OK_DONE}

# ---------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------

PRG_PATH=.

MAIN=main
TARGET=nvidioso

H_FOLDER=include
S_FOLDER=src

BASE=base
CONSTRAINTS=constraints
CORE=core
PARSER=FZ_parser
SEARCH=search

OBJECTS =  $(patsubst %.cpp,%.o, $(PRG_PATH)/$(MAIN).cpp)
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(S_FOLDER)/$(BASE)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(S_FOLDER)/$(CONSTRAINTS)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(S_FOLDER)/$(CORE)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(S_FOLDER)/$(PARSER)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(S_FOLDER)/$(SEARCH)/*.cpp))

# ---------------------------------------------------------------------
# Compiler options 
# ---------------------------------------------------------------------

CC = clang++

## debug, don't optimize 
# -m64: generate code for 64 bit env. Use -m32 for a 32 bit env.
# -DIL_STD -> #define IL_STD 1. Used to enable some STL features
CCOPT = -m64 -DIL_STD

## optimize
CCOPT += -O3 -std=c++0x 

vpath %.o obj

## lib dirs -L...
CCLNDIRS= 

## include dirs -I...
INCLDIRS =  -I$(PRG_PATH) 
INCLDIRS += -I$(PRG_PATH)/$(H_FOLDER) 
INCLDIRS += -I$(PRG_PATH)/$(H_FOLDER)/$(BASE)
INCLDIRS += -I$(PRG_PATH)/$(H_FOLDER)/$(CONSTRAINTS)
INCLDIRS += -I$(PRG_PATH)/$(H_FOLDER)/$(CORE)
INCLDIRS += -I$(PRG_PATH)/$(H_FOLDER)/$(PARSER)
INCLDIRS += -I$(PRG_PATH)/$(H_FOLDER)/$(SEARCH)

## Compiler Flags
CCFLAGS = $(CCOPT) $(INCLDIRS)

# Incrementally build your system.
all : 	nvidioso
	$(PRINT_DONE)

print : 
	echo $(OBJECTS)

nvidioso :	$(OBJECTS) 
	@$(CC) $(CCLNDIRS) -o $(TARGET) \
	$(OBJECTS:%=$(PRG_PATH)/obj/%)

# $<: the first prerequisite filename.
# $@: the target filename.
$(OBJECTS) : 	%.o : %.cpp
	@$(CC) -c $(CCFLAGS) $< -o $(PRG_PATH)/obj/$@
	$(PRINT_COMPILE)

#.PHONY : clean
clean:
	rm -f obj/*.o 	
	rm -f obj/*/*/*.o	
	rm -f $(TARGET)	