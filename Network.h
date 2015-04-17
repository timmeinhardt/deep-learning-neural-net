#pragma once

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

  vectorV getBiases() const;

  vectorM getWeights() const;

  // Methods
  void SGD(DataSet&, const int&, const int&, const double&, const DataSet& testData={});
    // non-const DataSet& because of random_shuffle()

  gsl_vector* feedforward(const gsl_vector*);
	
private:
  int numLayers;
	vector<int> sizes;
	vectorV biases;
	vectorM weights;

  // Methods
  int evaluate(const DataSet&);

  void trainWithMiniBatch(const DataSet&, const double&);

  pair<vectorV, vectorM> backprop(const pair<gsl_vector*, int>);

  pair<vectorV, vectorM> mallocPlaceholders();
  void freePlaceholders(pair<vectorV, vectorM>);
};