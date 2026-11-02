# Set directories path.
ROOT_DIR := .
BLD_DIR := $(ROOT_DIR)/build
EXM_DIR := $(ROOT_DIR)/example
INC_DIR := $(ROOT_DIR)/include
SRC_DIR := $(ROOT_DIR)/src

BIN_DIR := $(BLD_DIR)/bin
LIB_DIR := $(BLD_DIR)/lib
OBJ_DIR := $(BLD_DIR)/obj


# Set compiler flags.
CC := g++
CFLAGS := -std=c++23 -Wall -O3 -I $(INC_DIR)
LFLAGS := -lm


# Set archiver flags.
AR = ar
ARFLAGS = -rcs


# Set auxiliary variables.
RND := $(SRC_DIR)/random-generator.cpp $(INC_DIR)/random-generator.hpp
RND_OBJ := $(OBJ_DIR)/random-generator.o

SPARSE := $(SRC_DIR)/sparse-vector.cpp $(INC_DIR)/sparse-vector.hpp
SPARSE_OBJ := $(OBJ_DIR)/sparse-vector.o

SIMPLE := $(SRC_DIR)/simple-vector.cpp $(INC_DIR)/simple-vector.hpp
SIMPLE_OBJ := $(OBJ_DIR)/simple-vector.o

DATA := $(SRC_DIR)/data-set.cpp $(INC_DIR)/data-set.hpp
DATA_OBJ := $(OBJ_DIR)/data-set.o

ALG_NAM := SDCA SDCA-perm SDCA-cyclic SGD
ALG_SRC := $(addprefix $(SRC_DIR)/, $(addsuffix .cpp, $(ALG_NAM)))
ALG_OBJ := $(addprefix $(OBJ_DIR)/, $(addsuffix .o, $(ALG_NAM)))
ALG_HDR := $(INC_DIR)/solver.hpp

LIB_NAM := stoch-opt
LIB_HDR := $(INC_DIR)/sparse-vector.hpp $(INC_DIR)/simple-vector.hpp $(INC_DIR)/data-set.hpp $(ALG_HDR)
LIB := $(LIB_DIR)/$(LIB_NAM).a

EXM_NAM := hinge-loss
EXM_OBJ := $(addprefix $(OBJ_DIR)/, $(addsuffix .o, $(EXM_NAM)))
EXM_BIN := $(addprefix $(BIN_DIR)/, $(EXM_NAM))
EXM_RUN := $(addprefix run-, $(EXM_NAM))

RUN := $(EXM_DIR)/run.cpp $(EXM_DIR)/run.hpp
RUN_HDR := $(EXM_DIR)/run.hpp


# Declare the default target.
default: all


# Set targets and their dependencies for stoch-opt library.
$(RND_OBJ): $(RND)
	$(CC) $(CFLAGS) -c $< -o $@

$(SPARSE_OBJ): $(SPARSE)
	$(CC) $(CFLAGS) -c $< -o $@

$(SIMPLE_OBJ): $(SIMPLE) $(SPARSE)
	$(CC) $(CFLAGS) -c $< -o $@

$(DATA_OBJ): $(DATA) $(SIMPLE) $(SPARSE)
	$(CC) $(CFLAGS) -c $< -o $@

$(ALG_OBJ): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(ALG_HDR) $(DATA) $(SIMPLE) $(SPARSE) $(RND)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(ALG_OBJ) $(DATA_OBJ) $(SIMPLE_OBJ) $(SPARSE_OBJ) $(RND_OBJ)
	$(AR) $(ARFLAGS) $@ $^


# Set targets and their dependencies for examples.
$(OBJ_DIR)/run.o: $(RUN) $(LIB_HDR)
	$(CC) $(CFLAGS) -c $< -o $@

$(EXM_OBJ): $(OBJ_DIR)/%.o: $(EXM_DIR)/%.cpp $(RUN) $(LIB_HDR)
	$(CC) $(CFLAGS) -c $< -o $@

$(EXM_BIN): $(BIN_DIR)/%: $(OBJ_DIR)/%.o $(OBJ_DIR)/run.o $(LIB)
	$(CC) $(LFLAGS) $^ -o $@


# Set PHONY targets and their recipies.
.PHONY: default all clean $(ALG_NAM)

all: $(EXM_BIN)

clean:
	rm -f $(OBJ_DIR)/* $(BIN_DIR)/* $(LIB_DIR)/*

$(ALG_NAM): %: $(OBJ_DIR)/%.o
