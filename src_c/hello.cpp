#include <cstdio>

namespace {

void __attribute__((constructor)) hello() {
    puts("*** lljbash's par_ilu0_gmres_c.so loaded ***");
}

}
