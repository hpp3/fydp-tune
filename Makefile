CC=g++
CFLAGS=-std=c++11 -Wall -L./ -lcltune
DEPS= cpack.h packed_array_impl.h packed_2d_array_impl.h packed_array_section.h packed_array.h
OBJ = main.o json11.o packed_array_config.o

%.o: %.c $(DEPS)
		$(CC) -c -o $@ $< $(CFLAGS)

tuner: $(OBJ)
		g++ -o $@ $^ $(CFLAGS)
