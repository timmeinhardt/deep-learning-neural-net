#pragma once

#include <iostream>
#include <vector>
#include <math.h>

#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

using namespace std;
typedef vector<pair<gsl_vector*, int> > DataSet;
typedef vector<gsl_vector*> vectorV;
typedef vector<gsl_matrix*> vectorM;

gsl_vector* MatrixVectorMultiAndSum(const gsl_matrix*, const gsl_vector*, const gsl_vector*);
int gsl_matrix_mul_for_vectors(gsl_matrix*, const gsl_vector*, const gsl_vector*);
int gsl_matrix_mul(gsl_matrix*, const gsl_matrix*);

gsl_vector* SigmoidVectorized(gsl_vector*);
gsl_vector* SigmoidPrimeVectorized(gsl_vector*);

gsl_rng* GetGslRng();
gsl_vector* RandomGaussianGslVector(const gsl_rng*, const int);
gsl_matrix* RandomGaussianGslMatrix(const gsl_rng*, const int, const int);

DataSet BuildDataSet(const int, const int);

void PrintVector(const gsl_vector*);
void PrintImageVector(const gsl_vector*);
void PrintMatrix(const gsl_matrix*);
gsl_vector* CopyOfGslVector(const gsl_vector* v);
