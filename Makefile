TBLIS_LIB := $(HOME)/libtblis/lib/libtblis.a
TBLIS_INC := $(HOME)/libtblis/include/tblis
TCI_INC := $(HOME)/libtblis/include
TCI_LIB := $(HOME)/libtblis/lib/libtci.a
SMALL_INC := $(HOME)/SMaLLFramework/include
SMALL_PLATFORM := $(HOME)/SMaLLFramework/include/small/platforms/reference
LIBTORCH_DIR := $(HOME)

CPP := g++
CPPFLAGS := -std=c++17 -g -I$(LIBTORCH_DIR)/libtorch/include/torch/csrc/api/include -I$(LIBTORCH_DIR)/libtorch/include -I$(TBLIS_INC) -I$(TCI_INC) -I$(SMALL_INC) -I$(SMALL_PLATFORM) -D_GLIBCXX_USE_CXX11_ABI=0
LDFLAGS := $(TBLIS_LIB) $(TCI_LIB) -latomic -lhwloc -fopenmp -pthread -L$(LIBTORCH_DIR)/libtorch/lib -Wl,-rpath=$(LIBTORCH_DIR)/libtorch/lib -ltorch_cpu -lc10

SRC_OBJS := ref.o

TEST_OBJS := small_tests.o

TBLIS_TEST_OBJ := small_tblis.o

all: driver

driver: $(SRC_OBJS)
	$(CPP) $(SRC_OBJS) -o driver.x $(LDFLAGS)

small_tests: $(TEST_OBJS)
	$(CPP) $(TEST_OBJS) -o small_tests.x $(LDFLAGS)

tblis_test: $(TBLIS_TEST_OBJ)
	$(CPP) $(TBLIS_TEST_OBJ) -o tblis_test.x $(LDFLAGS)	

%.o: %.cpp
	$(CPP) $(CPPFLAGS) -c $< -o $@

run: driver
	./driver.x

run_tests: small_tests
	./small_tests.x

run_tblis_test: tblis_test
	./tblis_test.x	

clean:
	rm -f *.o *~ core *.x
