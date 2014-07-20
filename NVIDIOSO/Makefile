# Include a definition file
include ./make.inc

# ---------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------

MAIN=main
TARGET=nvidioso

OBJECTS =  $(patsubst %.cpp,%.o, $(PRG_PATH)/$(MAIN).cpp)
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(BASE)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(CONSTRAINTS)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(CORE)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(PARSER)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(SEARCH)/*.cpp))
OBJECTS += $(patsubst %.cpp,%.o,$(wildcard $(PRG_PATH)/$(NVIDIOSO_SRC)/$(EXCEPTION)/*.cpp))

# ---------------------------------------------------------------------
# Compiler options 
# ---------------------------------------------------------------------


vpath %.o obj

## lib dirs -L...
CCLNDIRS= 

## include dirs -I...
INCLDIRS =  -I$(PRG_PATH) 
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC) 
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(BASE)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(CONSTRAINTS)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(CORE)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(PARSER)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(SEARCH)
INCLDIRS += -I$(PRG_PATH)/$(NVIDIOSO_INC)/$(EXCEPTION)

## Compiler Flags
CCFLAGS = $(CCOPT) $(INCLDIRS)

# Incrementally build the system.
all : 	nvidioso
	$(PRINT_DONE)

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
	@rm -f obj/*.o 	
	@rm -f obj/*/*/*.o	
	@rm -f $(TARGET)	
	$(PRINT_CLEAN)	
