#include "misc.h"

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
  cout << endl;
	}
}

  
