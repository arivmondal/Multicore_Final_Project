CUDA_HOME ?=
ifeq ($(strip $(CUDA_HOME)),)
NVCC ?= nvcc
CUDA_LIB_DIR ?=
else
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_LIB_DIR ?= $(firstword $(wildcard $(CUDA_HOME)/lib64) $(wildcard $(CUDA_HOME)/lib/x64))
endif

CXX ?= g++
HOST_CXX ?= $(CXX)

TARGET := lowrank_svd
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj

CPPFLAGS := -Isrc
CXXFLAGS := -O3 -std=c++14 $(CPPFLAGS)
NVCCFLAGS := -O3 --std c++14 $(CPPFLAGS)
NVCC_HOST_FLAG := $(if $(strip $(HOST_CXX)),-ccbin $(HOST_CXX))
LDLIBS := $(if $(CUDA_LIB_DIR),-L$(CUDA_LIB_DIR)) -lcusolver -lcublas -lcudart

.PHONY: all clean

all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR)/$(TARGET): $(OBJ_DIR)/main.cu.o $(OBJ_DIR)/io.cpp.o | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(NVCC_HOST_FLAG) $^ -o $@ $(LDLIBS)

$(OBJ_DIR)/main.cu.o: src/main.cu src/io.hpp | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(NVCC_HOST_FLAG) -c $< -o $@

$(OBJ_DIR)/io.cpp.o: src/io.cpp src/io.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(BUILD_DIR)
