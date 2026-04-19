CUDA_HOME ?=
ifeq ($(strip $(CUDA_HOME)),)
NVCC ?= nvcc
CUDA_LIB_DIR ?=
else
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_LIB_DIR ?= $(firstword $(wildcard $(CUDA_HOME)/lib64) $(wildcard $(CUDA_HOME)/lib/x64))
endif

ifeq ($(OS),Windows_NT)
TARGET := lowrank_svd.exe
CXX := cl
HOST_CXX := cl
CPPFLAGS := /Isrc
CXXFLAGS := /nologo /O2 /std:c++14 /EHsc $(CPPFLAGS)
NVCCFLAGS := -O3 --std c++14 -Xcompiler /EHsc -Isrc
define MAKE_DIR
if not exist "$(subst /,\,$1)" mkdir "$(subst /,\,$1)"
endef
define REMOVE_DIR
if exist "$(subst /,\,$1)" rmdir /s /q "$(subst /,\,$1)"
endef
define COMPILE_CPP
$(CXX) $(CXXFLAGS) /c $< /Fo$@
endef
else
TARGET := lowrank_svd
CXX ?= g++
HOST_CXX ?= $(CXX)
CPPFLAGS := -Isrc
CXXFLAGS := -O3 -std=c++14 $(CPPFLAGS)
NVCCFLAGS := -O3 --std c++14 $(CPPFLAGS)
define MAKE_DIR
mkdir -p "$1"
endef
define REMOVE_DIR
rm -rf "$1"
endef
define COMPILE_CPP
$(CXX) $(CXXFLAGS) -c $< -o $@
endef
endif

BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj

NVCC_HOST_FLAG := $(if $(strip $(HOST_CXX)),-ccbin $(HOST_CXX))
LDLIBS := $(if $(CUDA_LIB_DIR),-L$(CUDA_LIB_DIR)) -lcusolver -lcublas -lcudart

.PHONY: all clean

all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR)/$(TARGET): $(OBJ_DIR)/main.cu.o $(OBJ_DIR)/io.cpp.o | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(NVCC_HOST_FLAG) $^ -o $@ $(LDLIBS)

$(OBJ_DIR)/main.cu.o: src/main.cu src/io.hpp | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(NVCC_HOST_FLAG) -c $< -o $@

$(OBJ_DIR)/io.cpp.o: src/io.cpp src/io.hpp | $(OBJ_DIR)
	$(COMPILE_CPP)

$(BUILD_DIR):
	$(call MAKE_DIR,$@)

$(OBJ_DIR):
	$(call MAKE_DIR,$@)

clean:
	$(call REMOVE_DIR,$(BUILD_DIR))
