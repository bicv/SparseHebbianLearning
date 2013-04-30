MEX = mkoctfile --mex #mex
NRFDIR = ./nrf
MEXT = path_to_matlab/bin/mexsg

cgf.$(MEXT): cgf.c
	$(MEX) -I$(NRFDIR) -L$(NRFDIR) -lnrfopt  cgf.c
