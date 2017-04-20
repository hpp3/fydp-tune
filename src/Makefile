IDIR =../include
CC=g++
CXXFLAGS=-std=c++11 -Wall -L../lib -lcltune -I$(IDIR)
ODIR=obj
_DEPS= pack.h packed_array_impl.h packed_2d_array_impl.h packed_array_section.h packed_array.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))
OBJ = main.o json11.o packed_array_config.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CXXFLAGS)

tuner: $(OBJ)
	g++ -o $@ $^ $(CXXFLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 
