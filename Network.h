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

  vectorV getBiases() const;

  vectorM getWeights() const;

  // Methods
  void SGD(DataSet&, const int&, const int&, const double&, const DataSet& testData={});
    // non-const DataSet& - random_shuffle()
	
private:
  int numLayers;
	vector<int> sizes;
	vectorV biases;
	vectorM weights;

  // Methods
  int evaluate(const DataSet&);

  gsl_vector* feedforward(const gsl_vector*);

  void update_mini_batch(const DataSet&, const double&);
};

#endif