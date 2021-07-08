pp = g++-4.8
cf = -std=c++11 -O3 -fPIC -fopenmp -fexceptions -march=native -ffast-math -funroll-loops
cf ?= -std=c++11 -g -fPIC -fopenmp -fexceptions
sh = -shared
lf = -lrt -ldl -lnuma -lpthread

OBJS = ilusolver libILUSolver.so libnaive.so

all: $(OBJS)

ilusolver: demo.cc libILUSolver.so libCCSMatrix.so
	$(pp) $(cf) -o $@ $< -Wl,-rpath=. -L. -lILUSolver -lCCSMatrix $(lf)
libILUSolver.so: ILUSolver.cc ILUSolver.h CommonDef.h
	$(pp) $(cf) $(sh) -o $@ $< -Wl,-rpath=. -L. -lCCSMatrix $(lf)
libnaive.so: ILUSolver_naive.cc ILUSolver.h CommonDef.h
	$(pp) $(cf) $(sh) -o $@ $< -Wl,-rpath=. -L. -lCCSMatrix $(lf)
clean:
	-rm $(OBJS)
