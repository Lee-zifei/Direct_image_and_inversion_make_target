#! bin/sh

#MPI=/home/Software/Intel/impi/5.0.1.035/intel64
#CUDA=/usr/local/cuda-11.1
MPI=/Softwares/mpich
CUDA=/Softwares/cuda11.4

include ~/softwares/makeopt/makefile.opt



NVCC = $(CUDA)/bin/nvcc
MPICC = $(MPI)/bin/mpicc


INC = -I$(MPI)/include -I$(CUDA)/include
LIB = -L$(MPI)/lib -L$(CUDA)/lib64
LDFLAGS = -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX -lmpi -lrt -lpthread -lcudart -lcurand -lcufft -lfftw3 
CFILES = acousticobs_moffsetC.cpp
CUFILES = acousticobs_moffset.cu
EXECNAME = Obser

obs:
	$(NVCC) -w  -o $(EXECNAME) $(CFILES) $(CUFILES) $(INC) $(LDFLAGS) $(LIB)
	./Obser ./input/mar_234_663.dat

rINC = -I$(MPI)/include -I$(CUDA)/include
rLDFLAGS =-L$(CUDA)/lib64 -L$(MPI)/lib/
rLIB = -lm -lstdc++ -lrt -lpthread -lcurand -lcufft -lcudart 
rCFILES = acoustic_HLRTMC.cpp
rCUFILES = acoustic_HLRTMLc.cu
rOBJECTS = acoustic_HLRTMC.o acoustic_HLRTMLc.o 
rEXECNAME = RTM


rtm:
	$(MPICC)  -w -c $(rCFILES)
	$(NVCC) -w -c $(rCUFILES)
	$(MPICC) -o $(rEXECNAME) $(rOBJECTS) $(rINC) $(rLDFLAGS) $(rLIB)
	./RTM ./input/mar_234_663.dat




cFILES = direct_cut.cpp 
cOBJECT= direct_cut.o
cEXECNAME = CUT
cut:
	$(MPICC)  -w -c ${cFILES} 
	$(MPICC) -o $(cEXECNAME) $(cOBJECT) $(rINC) $(rLDFLAGS) $(rLIB)
	./CUT ./input/mar_234_663.dat

clean:
	rm -rf *.o $(EXECNAME) $(rEXECNAME) CUT forward_output forward_slice rtm_output rtm_slice image 
	mkdir rtm_slice rtm_output forward_slice forward_output image
