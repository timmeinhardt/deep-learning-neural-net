#include "Network.h"

//
// Constructors
//

Network::Network() {
	numLayers = 0;
	sizes = {};
}

Network::Network(vector<int> newSizes) {
	numLayers = newSizes.size();
	sizes = newSizes;
  gsl_rng* rng = GetGslRng();

	// setBiases
  for(vector<int>::iterator it = sizes.begin() + 1; it != sizes.end(); ++it) {
    int numNeurons = *it;
    biases.push_back( RandomGaussianGslVector(rng, numNeurons) );
	}

	// setWeights
  for(vector<int>::iterator it = sizes.begin(); it != sizes.end() - 1; ++it) {
  	int numNeurons = *(it+1);
  	int numNeuronsPreviewsLayer = *it;
  	weights.push_back( RandomGaussianGslMatrix(rng, numNeurons, numNeuronsPreviewsLayer) );
	}
}

Network::~Network() {
}

//
// Accessors
//

int Network::getNumLayers() const {
	return numLayers;
}

vector<int> Network::getSizes() const {
	return sizes;
}

vectorV Network::getBiases() const {
	return biases;
}

vectorM Network::getWeights() const {
	return weights;
}

//
// Class Methods
//

// TODO: REFACTOR
gsl_vector* Network::feedforward(const gsl_vector* activation) {
  vectorM::const_iterator itWeightsLayer = weights.begin();
  gsl_vector* a = gsl_vector_alloc(activation->size);
  gsl_vector_memcpy(a, activation);

  for(gsl_vector* biasesLayer: biases) {
    // aNew with j of weights matrix
    gsl_vector* aNew = gsl_vector_alloc((*itWeightsLayer)->size1);
    
    // product of weights matrix and activation vector
    gsl_blas_dgemv(CblasNoTrans, 1.0, *itWeightsLayer, a, 0.0, aNew);
    // sum biasesLayer vector and product result
    gsl_blas_daxpy(1.0, biasesLayer, aNew);
    // sigmoid on new vector
    SigmoidVectorized(aNew);

    // free a before pointer is overwritten
    gsl_vector_free(a);
    a = aNew;

    ++itWeightsLayer; 
  }

  return a;
}

void Network::SGD(DataSet& trainingData, const int& epochs, const int& miniBatchSize, const double& eta, const DataSet& testData) {

  int nTestData = testData.size();

  // training epochs
  for (int i=1; i<=epochs; i++){
    random_shuffle( trainingData.begin(), trainingData.end() );

    for (DataSet::iterator startMiniBatch = trainingData.begin(); startMiniBatch != trainingData.end(); startMiniBatch = startMiniBatch + miniBatchSize) {
      // build miniBatch from trainingsData
      DataSet::iterator endMiniBatch = startMiniBatch + miniBatchSize;
      DataSet miniBatch(startMiniBatch, endMiniBatch);
      update_mini_batch(miniBatch, eta);
    }

    if (nTestData != 0) {
      cout << "Epoche " << i << " " << evaluate(testData) << "/" << nTestData << endl;
    } else {
      cout << "Epoche " << i << " completed." << endl;
    }
  }
}

int Network::evaluate(const DataSet& testData) {
  int result = 0;
  for (pair<gsl_vector*, int> pair: testData) {
    gsl_vector* activation = pair.first;
    int output = pair.second;
    if (gsl_vector_max_index(feedforward(activation)) == output) {
      result++;
    }
  }
  return result;
}


