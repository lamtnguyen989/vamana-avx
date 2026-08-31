# Compiling information
CFLAGS  ?= -O3 -flto -march=native
LINK_FLAG = -lm
SRC_DIR = src
BUILD_DIR = build
CC := mpicc

# Spack information
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

# Test compile for now
main: $(SRC_DIR)/main.c
	$(SETUP_SPACK)
	$(CC) $(CFLAGS) $< -o $(BUILD_DIR)/$@ $(LINK_FLAG)

# Training PQ codebook binary
train: $(SRC_DIR)/train_pq.c
	$(SETUP_SPACK)
	$(CC) $(CFLAGS) $< -o $(BUILD_DIR)/$@ $(LINK_FLAG)

clean:
	rm -rf $(BUILD_DIR)

