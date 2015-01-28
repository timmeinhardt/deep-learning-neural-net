#include "Network.h"

//
// Constructors
//

Network::Network() {
	numLayers = 0;
	sizes = {};
}

Network::Network(vector<int> sizes) {
	numLayers = sizes.size();
	sizes = sizes;

	// setBiases
  for(vector<int>::iterator it = sizes.begin() + 1; it != sizes.end(); ++it) {
    int numNeurons = *it;
    biases.push_back( RandomGaussianGslVector(numNeurons) );
	}

	// setWeights
  for(vector<int>::iterator it = sizes.begin(); it != sizes.end() - 1; ++it) {
  	int numNeurons = *(it+1);
  	int numNeuronsPreviewsLayer = *it;
  	weights.push_back( RandomGaussianGslMatrix(numNeurons, numNeuronsPreviewsLayer) );
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

vector<gsl_vector*> Network::getBiases() const {
	return biases;
}

vector<gsl_matrix*> Network::getWeights() const {
	return weights;
}

//
// Class Methods
//

// TODO: REFACTOR
gsl_vector* Network::feedforward(const gsl_vector* activation) {
  vector<gsl_matrix*>::const_iterator itWeightsLayer = weights.begin();
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
      //cout << miniBatch.size() << endl;
    }

    if (nTestData != 0) {
      cout << "Epoche " << i << " " << evaluate(testData) << "/" << nTestData << endl;
    } else {
      cout << "Epoche " << i << " completed." << endl;
    }
  }
}

int Network::evaluate(const DataSet& trainingData) {
  return 2;
}


