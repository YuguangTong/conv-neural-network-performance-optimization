all: cnn cnnModule.so

cnn: src/cnn.c src/util.c src/main.c
	gcc -mavx -g -std=c99 src/cnn.c -lm -fopenmp -o cnn

cnnModule.so: src/cnn.c src/python.c src/util.c
	gcc -mavx -std=c99 -shared -fopenmp -fPIC -I/usr/include/python2.7 -o cnnModule.so src/python.c src/cnn.c

run: cnnModule.so
	@python cnn.py

benchmark: cnn
	@cd test ; ../cnn benchmark

benchmark-small: cnn
	@cd test ; ../cnn benchmark 600

benchmark-large: cnn
	@cd test ; ../cnn benchmark 2400

benchmark-huge: cnn
	@cd test ; ../cnn benchmark 12000

test: cnn
	@cd test ; bash run_test.sh

clean:
	rm cnn cnnModule.so

.PHONY: run clean benchmark benchmark-small benchmark-large benchmark-huge test
