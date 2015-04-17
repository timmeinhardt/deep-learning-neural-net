#include "misc.h"

gsl_vector* MatrixVectorMultiAndSum(const gsl_matrix* m, const gsl_vector* v1, const gsl_vector* v2) {
  gsl_vector* vNew = gsl_vector_alloc((m)->size1);
    
  gsl_blas_dgemv(CblasNoTrans, 1.0, m, v1, 0.0, vNew);
  gsl_vector_add(vNew, v2);
  return vNew;
}

int gsl_matrix_mul(gsl_matrix* m1, const gsl_matrix* m2) {
  double multi;

  for (size_t i = 0; i < m1->size1; i++) {
    for (size_t j = 0; j < m1->size2; j++) {
      multi = gsl_matrix_get(m1, i, j) * gsl_matrix_get(m2, i, j);
      gsl_matrix_set(m1, i, j, multi);
    }
  }
  return 0;
}

int gsl_matrix_mul_for_vectors(gsl_matrix* m, const gsl_vector* v1, const gsl_vector* v2) {
  double multi;

  for (int i = 0; i < m->size1; i++) {
    for (int j = 0; j < m->size2; j++)
    {
      multi = gsl_vector_get(v1, i) * gsl_vector_get(v2, j);
      gsl_matrix_set(m, i, j, multi);
    }
  }  
  return 0;
}

double Sigmoid(double z) {
  return 1.0/(1.0 + exp(-z));
}

gsl_vector* SigmoidVectorized(gsl_vector* v) {
  for (int i=0; i<v->size; i++){
  	double sigmoidEntry = Sigmoid(gsl_vector_get(v,i));
    gsl_vector_set(v, i, sigmoidEntry);
  }
  return v;
}

gsl_vector* SigmoidPrimeVectorized(gsl_vector* v) {
  for (int i=0; i<v->size; i++){
  	double sigmoidEntry = Sigmoid(gsl_vector_get(v,i));
  	double sigmoidPrimeEntry = sigmoidEntry*(1.0 - sigmoidEntry);
    gsl_vector_set(v, i, sigmoidPrimeEntry);
  }
  return v;
}

void PrintVector(const gsl_vector* v) {
  for (int i=0; i<v->size; i++){
    cout << gsl_vector_get(v, i) << endl;
  }
  cout << endl;
}

void PrintMatrix(const gsl_matrix* m) {
  for (size_t i = 0; i < m->size1; i++) {
    for (size_t j = 0; j < m->size2; j++) {
      cout << gsl_matrix_get(m, i, j) << " ";
    }
    cout << endl;
	}
  cout << endl;
}

DataSet BuildDataSet(const int setSize,const int numNeuronsInput) {
	gsl_rng* rng = GetGslRng();
	DataSet set;

	for(int i=0; i<setSize; i++) {
		set.push_back(pair<gsl_vector*, int>(RandomGaussianGslVector(rng, numNeuronsInput), i%10));
	}
	return set;
}

gsl_rng* GetGslRng() {
  gsl_rng* rng;
	int random_seed = (int)time(NULL);
	rng = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(rng, random_seed);

	return rng;
}

gsl_vector* RandomGaussianGslVector(const gsl_rng* rng, const int i) {
	gsl_vector* v = gsl_vector_alloc(i);

  for (int it = 0; it < i; it++) {
    gsl_vector_set(v, it, gsl_ran_gaussian(rng, 1.0));
  }
  return v;
}

gsl_matrix* RandomGaussianGslMatrix(const gsl_rng* rng, const int i, const int j) {
  gsl_matrix* m = gsl_matrix_alloc(i, j);

  for (int it = 0; it < i; it++) {
  	for (int jt = 0; jt < j; jt++)
  	{
      gsl_matrix_set(m, it, jt, gsl_ran_gaussian(rng, 1.0));
  	}
  }
  return m;
}

gsl_vector* CopyOfGslVector(const gsl_vector* v) {
  gsl_vector* copy = gsl_vector_alloc(v->size);
  gsl_vector_memcpy(copy, v);

  return copy;
}

  
