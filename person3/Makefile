# Makefile for Person 3: Fused Low-Rank GPU Image Filter

NVCC = nvcc

NVCC_FLAGS = -O3 -std=c++14

TARGET = low_rank_filter

SRC = low_rank_filter.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)
	@echo "Build complete! Run with ./$(TARGET)"

clean:
	rm -f $(TARGET)
	@echo "Cleaned up build files."