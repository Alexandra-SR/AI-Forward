train:
	g++ train.cpp -std=c++17 -o train.out
	./train.out

test:
	g++ test.cpp -std=c++17 -o test.out
	./test.out