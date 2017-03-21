#include <vector>
#include <iostream>
#include <fstream>
#include <utility>
#include <algorithm>

#include "packed_2d_array_impl.h"

using namespace std;

int main() {
	int bitsize = 32;

    Packed2DArrayImpl a("test", bitsize, 10, 10, true);
    //Packed2DArrayImpl a("test", bitsize, 10, 10, false);

    for(int i=0; i<10; i++) {
		for (int j = 0; j < 10; j++) {
        	a.set(i, j, i);
		}
    }

    cout<<"size: "<<a.size()<<endl;
    cout<<"physical size: "<<a.physical_size()<<endl;
    cout<<"capacity: "<<a.capacity()<<endl;
    cout<<"physical capacity: "<<a.physical_capacity()<<endl;

    // read the file
    int group_size = 32;
    string header = a.generateOpenCLCode(false, group_size);
	cout << header << endl;

    // Creates data structures
    int32_t *cells = a.getCells();
	for (int i = 0; i < 100; i++) {
		cout << cells[i] << '\t';
	}
}
