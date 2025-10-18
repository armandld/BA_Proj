#include <pybind11/pybind11.h>

int add(int i, int j) { return i + j; }

PYBIND11_MODULE(hello, m) {
    m.def("add", &add, "Add two numbers");
}
