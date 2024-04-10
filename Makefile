TBLIS_LIB  := $(HOME)/libtblis/lib/libtblis.a
TBLIS_LIB_2  := $(HOME)/libtblis/lib/libtci.a
TBLIS_INC  := $(HOME)/libtblis/include/tblis
TCI_INC := $(HOME)/libtblis/include

CPP := g++
CPPFLAGS := -std=c++17 -I$(HOME)/libtorch/include/torch/csrc/api/include -I$(HOME)/libtorch/include -I$(TBLIS_INC) -I$(TCI_INC) -D_GLIBCXX_USE_CXX11_ABI=0
LDFLAGS := -L$(HOME)/libtorch/lib -Wl,-rpath=$(HOME)/libtorch/lib -ltorch_cpu -lc10

SRC_OBJS := ref.cpp

all: driver

driver: $(SRC_OBJS)
	$(CPP) $(CPPFLAGS) -L$(TBLIS_LIB) -L$(TBLIS_LIB_2) -o driver.x $^ $(LDFLAGS)

run: driver
	./driver.x 

clean:
	rm -f *.o *~ core *.x


# https://discuss.pytorch.org/t/how-do-i-to-create-a-new-project-in-c-to-run-libtorch-without-cmake/79046
