CPP = g++

# Standard
FLAGS = -std=gnu++11
# Optimization
FLAGS += -Ofast -flto -march=native -funroll-loops
# Warning
FLAGS += -Wall -Wno-unused-result
# Msic
FLAGS += -lm -pthread
# FLAGS += -g -O0


all: learn_suc

learn_suc : learn_suc.cpp
	$(CPP) learn_suc.cpp -o learn_suc $(FLAGS)

clean: 
	rm learn_suc
