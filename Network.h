#ifndef Network_H
#define Network_H

#include "misc.h"

using namespace std;

class Network {
public:
	// Default Constructor
	Network();

	// Overload Constructor
  Network(vector<int>);

  // Destructor
  ~Network();

  // Accessors 
  int getNumLayers() const;

  vector<int> getSizes() const;

  vector<gsl_vector*> getBiases() const;

  vector<gsl_matrix*> getWeights() const;

  // Methods
  gsl_vector* feedforward(const gsl_vector*);

  void SGD(DataSet&, const int&, const int&, const double&, const DataSet& testData={});
    // non-const DataSet& - random_shuffle()
	
private:
  int numLayers;
	vector<int> sizes;
	vector<gsl_vector*> biases;
	vector<gsl_matrix*> weights;

  // Methods
  int evaluate(const DataSet&);
};

#endif