CXX ?= g++
CXXFLAGS ?= -std=c++20 -O3 -mavx2 -mfma -Wall -Wextra -Iinclude -pthread
LDFLAGS ?= -pthread

BUILD_DIR = build
BIN_DIR = bin

CORE_SRCS = src/engine/distance.cpp \
            src/engine/hnsw_index.cpp \
            src/engine/ivf_index.cpp \
            src/storage/wal.cpp \
            src/cluster/consistent_hash.cpp

CORE_OBJS = $(patsubst src/%.cpp, $(BUILD_DIR)/%.o, $(CORE_SRCS))

all: server test bench demo

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

server: $(CORE_OBJS) $(BUILD_DIR)/server/main.o
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $(BIN_DIR)/vectordb $(LDFLAGS)

$(BUILD_DIR)/server/main.o: src/server/main.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(CORE_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) tests/test_distance.cpp $(CORE_OBJS) -o $(BIN_DIR)/test_distance $(LDFLAGS)
	$(CXX) $(CXXFLAGS) tests/test_hnsw.cpp $(CORE_OBJS) -o $(BIN_DIR)/test_hnsw $(LDFLAGS)
	$(CXX) $(CXXFLAGS) tests/test_ivf.cpp $(CORE_OBJS) -o $(BIN_DIR)/test_ivf $(LDFLAGS)
	@echo "================ Running Tests ================"
	./$(BIN_DIR)/test_distance
	./$(BIN_DIR)/test_hnsw
	./$(BIN_DIR)/test_ivf

bench: $(CORE_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) bench/bench_distance.cpp $(CORE_OBJS) -o $(BIN_DIR)/bench_distance $(LDFLAGS)
	@echo "================ Running Distance Benchmark ================"
	./$(BIN_DIR)/bench_distance

demo: $(CORE_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) examples/demo_semantic_search.cpp $(CORE_OBJS) -o $(BIN_DIR)/demo $(LDFLAGS)
	@echo "================ Running Semantic Search Demo ================"
	./$(BIN_DIR)/demo

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) data/

.PHONY: all server test bench demo clean
