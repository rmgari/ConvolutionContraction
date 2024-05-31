TBLIS_LIB := $(HOME)/libtblis/lib/libtblis.a
TBLIS_INC := $(HOME)/libtblis/include/tblis
TCI_INC := $(HOME)/libtblis/include
TCI_LIB := $(HOME)/libtblis/lib/libtci.a
LIBTORCH_DIR := $(HOME)

CPP := g++
CPPFLAGS := -std=c++17 -I$(LIBTORCH_DIR)/libtorch/include/torch/csrc/api/include -I$(LIBTORCH_DIR)/libtorch/include -I$(TBLIS_INC) -I$(TCI_INC) -D_GLIBCXX_USE_CXX11_ABI=0
LDFLAGS := $(TBLIS_LIB) $(TCI_LIB) -latomic -lhwloc -fopenmp -pthread -L$(LIBTORCH_DIR)/libtorch/lib -Wl,-rpath=$(LIBTORCH_DIR)/libtorch/lib -ltorch_cpu -lc10

SRC_OBJS := ref.o

all: driver

driver: $(SRC_OBJS)
	$(CPP) $(SRC_OBJS) -o driver.x $(LDFLAGS)

%.o: %.cpp
	$(CPP) $(CPPFLAGS) -c $< -o $@

run: driver
	./driver.x

clean:
	rm -f *.o *~ core *.x
