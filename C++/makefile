# Makefile

pcaProg = bcPCA
pcaObjs = bayesClusteringPCA.o bayesClusteringPCA_core.o bcPCA_dataHandler.o dataUtils.o mathUtils.o rand.o

mgProg = bcMixGauss
mgObjs = MG_bayesClustering.o MG_bayesClustering_core.o MG_dataHandler.o MG_dataUtils.o mathUtils.o rand.o

INCLUDES = /usr/local/include
LIBS = /usr/local/lib

# for gcc
#CC = g++
#CCFLAGS = -O3 -fopenmp
#CXXFLAGS = -O3 -fopenmp
#LDFLAGS = -lm -lgsl -lgslcblas -fopenmp

# for clang
CC = clang++
#CCFLAGS = -O3
CXXFLAGS = -O3 -I$(INCLUDES)
LDFLAGS = -lgsl -lgslcblas -L$(LIBS)

# for intel compiler
#CC = icc
#CCFLAGS = -O3 -openmp
#CXXFLAGS = -O3 -openmp
#LDFLAGS = -lm -lgsl -lgslcblas -static-intel

.SUFFIXES: .c .cpp .o

$(program) : $(objs)
	$(CC) $^ $(LDFLAGS) -o $(program)

pca : $(pcaProg)
$(pcaProg) : $(pcaObjs)
	$(CC) $^ $(LDFLAGS) -o $(pcaProg)

gauss : $(mgProg)
$(mgProg) : $(mgObjs)
	$(CC) $^ $(LDFLAGS) -o $(mgProg)

.c.cpp.o:
	$(CC) $(CCFLAGS) -c $<

.PHONY: clean
clean:
	$(RM) $(pcaProg) $(pcaObjs) $(mgProg) $(mgObjs)

bayesClusteringPCA.o : bayesClusteringPCA.h bayesClusteringPCA_core.h dataUtils.h mathUtils.h rand.h
bayesClusteringPCA_core.o : bayesClusteringPCA_core.h dataUtils.h mathUtils.h
bcPCA_datahandler.o : dataUtils.h
dataUtils.o : dataUtils.h

MG_bayesClustering.o : MG_bayesClustering.h MG_bayesClustering_core.h MG_dataUtils.h mathUtils.h rand.h
MG_bayesClustering_core.o : MG_bayesClustering_core.h MG_dataUtils.h mathUtils.h
MG_datahandler.o : MG_dataUtils.h
MG_dataUtils.o : MG_dataUtils.h

mathUtils.o : mathUtils.h
rand.o : rand.h

#
