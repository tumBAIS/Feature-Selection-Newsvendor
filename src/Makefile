# Makefile ready for Linux 
SYSTEM     = x86-64_linux
LIBFORMAT  = static_pic

CPLEXVERSION = CPLEX_Studio1210

CPLEXDIR      = /opt/ibm/ILOG/$(CPLEXVERSION)/cplex
CONCERTDIR    = /opt/ibm/ILOG/$(CPLEXVERSION)/concert

CCC = g++ 

CCOPT = -m64 -O3 -w -fPIC -fno-strict-aliasing -fexceptions -DNDEBUG -DIL_STD -std=c++14

CPLEXLIBDIR   = $(CPLEXDIR)/lib/$(SYSTEM)/$(LIBFORMAT)
CONCERTLIBDIR = $(CONCERTDIR)/lib/$(SYSTEM)/$(LIBFORMAT)
CPLEXINCDIR   = $(CPLEXDIR)/include
CONCERTINCDIR = $(CONCERTDIR)/include
CCFLAGS = $(CCOPT) -I$(CONCERTINCDIR) -I$(CPLEXINCDIR)
CCLNDIRS  = -L$(CPLEXLIBDIR) -L$(CONCERTLIBDIR)
CCLNFLAGS = -lconcert -lilocplex -lcplex -lm -pthread -ldl
TARGETDIR = .

OBJS2 = \
	$(TARGETDIR)/commandline.o \
	$(TARGETDIR)/main.o \
	$(TARGETDIR)/Pb_Data.o \
	$(TARGETDIR)/Solution.o \
	$(TARGETDIR)/Solution_ERM.o \
	$(TARGETDIR)/Solution_HO.o \
	$(TARGETDIR)/Solver.o \
	$(TARGETDIR)/Solver_ERM.o \
	$(TARGETDIR)/Solver_HO.o \

# ------------------------------------------------------------
EXEC=newsvendor-features

all:$(EXEC)


$(EXEC): $(OBJS2)
	$(CCC) $(CCFLAGS) $(CCLNDIRS) -o $(EXEC) $(OBJS2) $(CCLNFLAGS) 

$(TARGETDIR)/main.o: main.cpp
	$(CCC) $(CCFLAGS) $(LIBS)  $(CPPFLAGS) -c main.cpp -o $(TARGETDIR)/main.o  	

$(TARGETDIR)/commandline.o: commandline.h commandline.cpp
	$(CCC) $(CCFLAGS)  $(CPPFLAGS) -c commandline.cpp -o $(TARGETDIR)/commandline.o  	

$(TARGETDIR)/Pb_Data.o: Pb_Data.h Pb_Data.cpp
	$(CCC) $(CCFLAGS)  $(CPPFLAGS) -c Pb_Data.cpp -o $(TARGETDIR)/Pb_Data.o   	

$(TARGETDIR)/Solution.o: Solution.h Solution.cpp
	$(CCC) $(CCFLAGS)  $(CPPFLAGS) -c Solution.cpp -o $(TARGETDIR)/Solution.o  	

$(TARGETDIR)/Solution_ERM.o: Solution.h Solution_ERM.cpp
	$(CCC) $(CCFLAGS)  $(CPPFLAGS) -c Solution_ERM.cpp -o $(TARGETDIR)/Solution_ERM.o  	

$(TARGETDIR)/Solution_HO.o: Solution.h Solution_HO.cpp
	$(CCC) $(CCFLAGS)  $(CPPFLAGS) -c Solution_HO.cpp -o $(TARGETDIR)/Solution_HO.o  	

$(TARGETDIR)/Solver.o: Solver.h Solver.cpp
	$(CCC) $(CCFLAGS)  $(CPPFLAGS) -c Solver.cpp -o $(TARGETDIR)/Solver.o  	

$(TARGETDIR)/Solver_ERM.o: Solver.h Solver_ERM.cpp
	$(CCC) $(CCFLAGS)  $(CPPFLAGS) -c Solver_ERM.cpp -o $(TARGETDIR)/Solver_ERM.o  	

$(TARGETDIR)/Solver_HO.o: Solver.h Solver_HO.cpp
	$(CCC) $(CCFLAGS)  $(CPPFLAGS) -c Solver_HO.cpp -o $(TARGETDIR)/Solver_HO.o  	

clean:
	rm -rf *.o $(EXEC)
