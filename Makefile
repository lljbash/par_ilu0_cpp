pp = g++
#cf = -std=c++11 -O3 -DNDEBUG -fPIC -fopenmp -fexceptions -march=native -ffast-math -funroll-loops
cf ?= -std=c++11 -g -fPIC -fopenmp #-fsanitize=address -fno-omit-frame-pointer
cf += -D_GLIBCXX_USE_CXX11_ABI=0
sh = -shared
lf = -lrt -ldl -lnuma -lpthread

OBJS = ilusolver libILUSolver.so libnaive.so libpar_ilu0_c.so
HEADERS = ILUSolver.h CommonDef.h scope_guard.h subtree.h

all: $(OBJS)

ilusolver: demo.cc libILUSolver.so libCCSMatrix.so
	$(pp) $(cf) -o $@ $< -Wl,-rpath=. -L. -lILUSolver -lCCSMatrix $(lf)
libpar_ilu0_c.so: par_ilu0_c.cpp ILUSolver.cc par_ilu0_c.h itsol/globheads.h $(HEADERS)
	$(pp) $(cf) -DCHECK_DIAG $(sh) -o $@ $< ILUSolver.cc $(lf)
libILUSolver.so: ILUSolver.cc $(HEADERS)
	$(pp) $(cf) $(sh) -o $@ $< -Wl,-rpath=. -L. -lCCSMatrix $(lf)
libnaive.so: ILUSolver_naive.cc $(HEADERS)
	$(pp) $(cf) $(sh) -o $@ $< -Wl,-rpath=. -L. -lCCSMatrix $(lf)
clean:
	-rm $(OBJS)
