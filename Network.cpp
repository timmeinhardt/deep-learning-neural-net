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
  for(vector<int>::iterator it = sizes.begin() + 1; it != sizes.end(); ++it) {
  	int numNeurons = *it;
  	int numNeuronsPreviewsLayer = *(it - 1);
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

//
// Feedforward activation through neural net
// TODO: REFACTOR
//
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

//
// Start stochastic gradient descent
//
void Network::SGD(DataSet& trainingData, const int& epochs, const int& miniBatchSize, const double& eta, const DataSet& testData) {

  int nTestData = testData.size();

  // training epochs
  for (int i=1; i<=epochs; i++){
    random_shuffle( trainingData.begin(), trainingData.end() );

    // split trainingData and train with miniBatch
    for (DataSet::iterator startMiniBatch = trainingData.begin(); startMiniBatch != trainingData.end(); startMiniBatch = startMiniBatch + miniBatchSize) {
      // build miniBatch from trainingsData
      DataSet::iterator endMiniBatch = startMiniBatch + miniBatchSize;
      DataSet miniBatch(startMiniBatch, endMiniBatch);

      train_with_mini_batch(miniBatch, eta);
    }

    // evaluate test data and print result for each epoche
    if (nTestData != 0) {
      cout << "Epoche " << i << " " << evaluate(testData) << "/" << nTestData << endl;
    } else {
      cout << "Epoche " << i << " completed." << endl;
    }
  }
}

//
// Evaluate test data
//
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

//
// Train neural net with trainings data
//
void Network::train_with_mini_batch(const DataSet& miniBatch, const double& eta) {
  // set zero biases as nabla start
  vectorV nabla_biases;
  for(vector<int>::iterator it = sizes.begin() + 1; it != sizes.end(); ++it) {
    int numNeurons = *it;
    nabla_biases.push_back( gsl_vector_alloc (numNeurons) );
  }

  // set zero weights as nabla start
  vectorM nabla_weights;
  for(vector<int>::iterator it = sizes.begin() + 1; it != sizes.end(); ++it) {
    int numNeurons = *it;
    int numNeuronsPreviewsLayer = *(it - 1);
    nabla_weights.push_back( gsl_matrix_alloc(numNeurons, numNeuronsPreviewsLayer) );
  }
  
  // iterate over training pairs and update biases and weights
  for (pair<gsl_vector*, int> trainingPair: miniBatch) {
    pair<vectorV, vectorM> delta_nablas = backprop(trainingPair);

    // add delta_nabla_biases from backprob(trainingsPair) to nabla_biases
    vectorV::const_iterator it_nabla_biases = nabla_biases.begin();
    for(gsl_vector* delta_nabla_bias: delta_nablas.first) { 
      gsl_vector_add(*it_nabla_biases, delta_nabla_bias);
      ++it_nabla_biases;
    }

    // add delta_nabla_weights from backprob(trainingsPair) to nabla_weights
    vectorM::const_iterator it_nabla_weights = nabla_weights.begin();
    for(gsl_matrix* delta_nabla_weight: delta_nablas.second) { 
      gsl_matrix_add(*it_nabla_weights, delta_nabla_weight);
      ++it_nabla_weights;
    }
  }

  // free gsl nablas???
}

//
// Back propagation
//
pair<vectorV, vectorM> Network::backprop(const pair<gsl_vector*, int>) {
  // set zero biases as placeholder
  vectorV nabla_biases;
  for(vector<int>::iterator it = sizes.begin() + 1; it != sizes.end(); ++it) {
    int numNeurons = *it;
    nabla_biases.push_back( gsl_vector_alloc (numNeurons) );
  }

  // set zero weights as placeholder
  vectorM nabla_weights;
  for(vector<int>::iterator it = sizes.begin(); it != sizes.end() - 1; ++it) {
    int numNeurons = *(it+1);
    int numNeuronsPreviewsLayer = *it;
    nabla_weights.push_back( gsl_matrix_alloc(numNeurons, numNeuronsPreviewsLayer) );
  }

  return pair<vectorV, vectorM>(nabla_biases, nabla_weights);
}


