# Compiling information
CFLAGS  ?= -O3 -flto -march=native
LINK_FLAG = -lm -fopenmp
SRC_DIR = src
BUILD_DIR = build
MPICC := mpicc

METRIC_IMPL ?= -DL2_IMPLEMENTATION

# Spack information
# How likely is it that people use this name for one of their spack env name?
SPACK_ENV_NAME ?= vamana-avx
SPACK_ENV_YAML ?= $(CURDIR)/environments/spack.yaml

SHELL := /bin/bash

.ONESHELL:

# Disgusting bolted-on way to setup spack environment (globally)
define SETUP_SPACK
	source $(SPACK_ROOT)/share/spack/setup-env.sh

	# Create the environment if it doesn't exist.
	if ! spack env list | grep -qE '^[[:space:]]*$(SPACK_ENV_NAME)([[:space:]]|$$)'; then
		echo "Creating Spack environment: $(SPACK_ENV_NAME)"
		spack env create $(SPACK_ENV_NAME) $(SPACK_ENV_YAML)
	else
		echo "Spack environment exists: $(SPACK_ENV_NAME)"
	fi

	# Make sure the repo's spack.yaml is the source of truth.
	spack env activate $(SPACK_ENV_NAME)
	spack install

	mkdir -p $(BUILD_DIR)
endef

# Training PQ codebook binary
train: $(SRC_DIR)/train_pq.c
	$(SETUP_SPACK)
	$(MPICC) $(CFLAGS) $< -o $(BUILD_DIR)/$@ $(LINK_FLAG) $(METRIC_IMPL)

# Encode PQ on data shard after training a codebook binary
encode: $(SRC_DIR)/encode_pq.c
	$(CC) $(CFLAGS) $< -o $(BUILD_DIR)/$@ $(LINK_FLAG) $(METRIC_IMPL)

clean:
	rm -rf $(BUILD_DIR)

