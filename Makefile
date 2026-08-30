CFLAGS  ?= -O3 -flto -march=native
LINK_FLAG = -lm
SRC_DIR = src
BUILD_DIR = build

# Test compile for now
main: $(SRC_DIR)/main.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $< -o $(BUILD_DIR)/$@ $(LINK_FLAG)

clean:
	rm -rf $(BUILD_DIR)

