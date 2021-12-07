CXX=clang++
INCLUDES=-Iincludes/
CXXFLAGS=-std=c++2a -g -fstandalone-debug -Wall -Wextra -Werror -pedantic $(INCLUDES)

exec: bin/exec
tests: bin/tests

bin/exec: ./src/driver.cc ./src/dataframe.cc ./src/mlp.cc
	$(CXX) $(CXXFLAGS) $^ -o $@

bin/tests: ./tests/tests.cc ./src/dataframe.cc ./src/mlp.cc
	$(CXX) $(CXXFLAGS) $^ -o $@

.DEFAULT_GOAL := tests
.PHONY: clean exec tests

clean:
	rm -fr bin/* obj/*
