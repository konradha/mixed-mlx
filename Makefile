DEVELOPER_DIR ?= /Applications/Xcode.app/Contents/Developer
XCRUN := env DEVELOPER_DIR=$(DEVELOPER_DIR) xcrun
CXX := $(XCRUN) clang++
METAL := $(XCRUN) metal
METALLIB := $(XCRUN) metallib
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -Wpedantic -Werror -Wno-gnu-anonymous-struct -DACCELERATE_NEW_LAPACK -Ithird_party/metal-cpp
METALFLAGS := -std=metal4.1 -O3 -fno-fast-math
FRAMEWORKS := -framework Accelerate -framework Foundation -framework Metal -framework MetalPerformanceShaders
BUILD := build
RESULTS ?= artifacts/gpu_staging_final
WARMUPS ?= 50
EVALUATIONS ?= 100

BASELINE_SOURCES := 1_baseline/main.cpp 1_baseline/spd.cpp 1_baseline/metal.cpp 1_baseline/metal_spd.cpp
BASELINE_OBJECTS := $(patsubst %.cpp,$(BUILD)/%.o,$(BASELINE_SOURCES)) $(BUILD)/1_baseline/mps_matmul.o
FORMAT_SOURCES := 2_formats/main.cpp 2_formats/formats.cpp 2_formats/metal.cpp
FORMAT_OBJECTS := $(patsubst %.cpp,$(BUILD)/%.o,$(FORMAT_SOURCES)) $(BUILD)/2_formats/tensor_layout.o
SPECTRAL_SOURCES := 3_spectral/main.cpp 3_spectral/spectral.cpp 3_spectral/metal.cpp 3_spectral/lowbit_matmul.cpp
SPECTRAL_OBJECTS := $(patsubst %.cpp,$(BUILD)/%.o,$(SPECTRAL_SOURCES)) $(BUILD)/3_spectral/mps_matmul.o
PRODUCT_SOURCES := 3_spectral/product_benchmark.cpp 3_spectral/metal.cpp 3_spectral/mps_matmul.cpp 3_spectral/lowbit_matmul.cpp
PRODUCT_OBJECTS := $(patsubst %.cpp,$(BUILD)/%.o,$(PRODUCT_SOURCES))
MIXED_SOURCES := $(wildcard src/*.cpp)
CHECK_OBJECTS := $(BUILD)/tests/matrix_codec.o $(BUILD)/3_spectral/metal.o $(BUILD)/3_spectral/lowbit_matmul.o $(BUILD)/3_spectral/spectral.o $(BUILD)/3_spectral/mps_matmul.o
BASELINE_LIBRARY := $(BUILD)/1_baseline/kernels.metallib
FORMAT_LIBRARY := $(BUILD)/2_formats/kernels.metallib
SPECTRAL_LIBRARY := $(BUILD)/3_spectral/kernels.metallib
PRODUCT_LIBRARY := $(BUILD)/3_spectral/products.metallib

.PHONY: all baseline formats spectral products mixed quick check bench toolchain-check clean

all: baseline formats spectral products mixed

baseline: $(BUILD)/baseline

formats: $(BUILD)/formats

spectral: $(BUILD)/spectral

products: $(BUILD)/products

mixed: $(BUILD)/mixed_sqrt

quick: $(BUILD)/baseline $(BUILD)/formats $(BUILD)/spectral
	./$(BUILD)/baseline --quick --output artifacts/baseline_quick
	./$(BUILD)/formats --quick --output artifacts/formats_quick
	./$(BUILD)/spectral --quick --output artifacts/spectral_quick

check: export MTL_DEBUG_LAYER = 1
check: export MTL_SHADER_VALIDATION = 1
check: export MTL_SHADER_VALIDATION_REPORT_TO_STDERR = 1
check: export MTL_SHADER_VALIDATION_ABORT_ON_FAULT = 1
check: all $(BUILD)/matrix_codec_check
	./$(BUILD)/matrix_codec_check
	./$(BUILD)/mixed_sqrt --codec-check
	./$(BUILD)/mixed_sqrt --kernel-check
	./$(BUILD)/baseline --quick --output $(BUILD)/checks/baseline
	./$(BUILD)/formats --quick --output $(BUILD)/checks/formats
	./$(BUILD)/spectral --order 96 --warmups 1 --evaluations 1 --max-corrections 30 --output $(BUILD)/checks/spectral
	./$(BUILD)/products --quick --output $(BUILD)/checks/products.csv

bench: all
	./$(BUILD)/baseline --warmups $(WARMUPS) --evaluations $(EVALUATIONS) --output $(RESULTS)/baseline
	./$(BUILD)/formats --warmups $(WARMUPS) --evaluations $(EVALUATIONS) --output $(RESULTS)/formats
	./$(BUILD)/products --order 128 --warmups $(WARMUPS) --evaluations $(EVALUATIONS) --output $(RESULTS)/products/n128.csv
	./$(BUILD)/products --order 256 --warmups $(WARMUPS) --evaluations $(EVALUATIONS) --output $(RESULTS)/products/n256.csv
	./$(BUILD)/products --order 512 --warmups $(WARMUPS) --evaluations $(EVALUATIONS) --output $(RESULTS)/products/n512.csv
	./$(BUILD)/spectral --order 128 --warmups $(WARMUPS) --evaluations $(EVALUATIONS) --max-corrections 30 --output $(RESULTS)/spectral

toolchain-check:
	@test "$(DEVELOPER_DIR)" = "/Applications/Xcode.app/Contents/Developer" || { echo "Metal 4.1 requires the Xcode 27 developer directory."; exit 1; }
	@test "$$($(XCRUN) -sdk macosx --show-sdk-version)" = "27.0" || { echo "Metal 4.1 requires the macOS 27.0 SDK."; exit 1; }
	@case "$$($(XCRUN) metal --version)" in *32023.921*) ;; *) echo "Metal 4.1 requires metalfe 32023.921."; exit 1;; esac

$(BUILD)/baseline: $(BASELINE_OBJECTS) $(BASELINE_LIBRARY)
	$(CXX) $(CXXFLAGS) $(filter %.o,$^) $(FRAMEWORKS) -o $@

$(BUILD)/formats: $(FORMAT_OBJECTS) $(FORMAT_LIBRARY)
	$(CXX) $(CXXFLAGS) $(filter %.o,$^) $(FRAMEWORKS) -o $@

$(BUILD)/spectral: $(SPECTRAL_OBJECTS) $(SPECTRAL_LIBRARY)
	$(CXX) $(CXXFLAGS) $(filter %.o,$^) $(FRAMEWORKS) -o $@

$(BUILD)/products: $(PRODUCT_OBJECTS) $(PRODUCT_LIBRARY)
	$(CXX) $(CXXFLAGS) $(filter %.o,$^) $(FRAMEWORKS) -o $@

$(BUILD)/mixed_sqrt: $(MIXED_SOURCES) | toolchain-check
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -Isrc $(MIXED_SOURCES) $(FRAMEWORKS) -o $@

$(BUILD)/matrix_codec_check: $(CHECK_OBJECTS) $(SPECTRAL_LIBRARY)
	$(CXX) $(CXXFLAGS) $(filter %.o,$^) $(FRAMEWORKS) -o $@

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@


$(BUILD)/1_baseline/kernels.air: 1_baseline/kernels.metal | toolchain-check
	@mkdir -p $(dir $@)
	$(METAL) $(METALFLAGS) -c $< -o $@

$(BASELINE_LIBRARY): $(BUILD)/1_baseline/kernels.air
	$(METALLIB) $^ -o $@

$(BUILD)/2_formats/kernels.air: 2_formats/kernels.metal | toolchain-check
	@mkdir -p $(dir $@)
	$(METAL) $(METALFLAGS) -c $< -o $@

$(FORMAT_LIBRARY): $(BUILD)/2_formats/kernels.air
	$(METALLIB) $^ -o $@

$(BUILD)/3_spectral/codecs.air: 3_spectral/codecs.metal | toolchain-check
	@mkdir -p $(dir $@)
	$(METAL) $(METALFLAGS) -c $< -o $@

$(BUILD)/3_spectral/refine.air: 3_spectral/refine.metal | toolchain-check
	@mkdir -p $(dir $@)
	$(METAL) $(METALFLAGS) -c $< -o $@

$(BUILD)/3_spectral/lowbit_matmul.air: 3_spectral/lowbit_matmul.metal | toolchain-check
	@mkdir -p $(dir $@)
	$(METAL) $(METALFLAGS) -c $< -o $@

$(SPECTRAL_LIBRARY): $(BUILD)/3_spectral/codecs.air $(BUILD)/3_spectral/refine.air $(BUILD)/3_spectral/lowbit_matmul.air
	$(METALLIB) $^ -o $@

$(PRODUCT_LIBRARY): $(BUILD)/3_spectral/codecs.air $(BUILD)/3_spectral/lowbit_matmul.air
	$(METALLIB) $^ -o $@

clean:
	rm -rf $(BUILD)

-include $(BASELINE_OBJECTS:.o=.d) $(FORMAT_OBJECTS:.o=.d) $(SPECTRAL_OBJECTS:.o=.d) $(PRODUCT_OBJECTS:.o=.d) $(CHECK_OBJECTS:.o=.d)
