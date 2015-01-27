#ifndef MISC_H
#define MISC_H

#include <iostream>
#include <vector>
#include <math.h>

#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

using namespace std;
typedef vector<tuple<gsl_vector* , int>> DataSet;

gsl_vector* SigmoidVectorized(gsl_vector*);
gsl_vector* SigmoidPrimeVectorized(gsl_vector*);
void PrintVector(const gsl_vector*);
void PrintMatrix(const gsl_matrix*);

#endif
