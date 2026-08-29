CFLAGS  ?= -O3 -flto -march=native -lm
SRC_DIR = src
BUILD_DIR = build

# Test compile for now
main: $(SRC_DIR)/main.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $< -o $(BUILD_DIR)/$@

clean:
	rm -rf $(BUILD_DIR)
