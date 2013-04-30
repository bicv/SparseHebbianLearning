#include "mex.h"
#include <unistd.h>
mexFunction(int nlhs, mxArray *plhs[ ], int nrhs, const mxArray *prhs[ ])
{
  plhs[0] = mxCreateDoubleScalar((double) getpid());
}
