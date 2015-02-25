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

	// set random biases and weights 
  for(vector<int>::iterator it = sizes.begin() + 1; it != sizes.end(); ++it) {
    int numNeurons = *it;
    int numNeuronsPreviewsLayer = *(it - 1);
    
    biases.push_back( RandomGaussianGslVector(rng, numNeurons) );
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
  gsl_vector* a = gsl_vector_alloc(activation->size);
  gsl_vector_memcpy(a, activation);
  gsl_vector* aNew;

  vectorM::const_iterator itWeightsLayer = weights.begin();
  for(gsl_vector* biasesLayer: biases) {

    aNew = MatrixVectorMultiAndSum(*itWeightsLayer, a, biasesLayer);
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

  // iterate over training epochs, train network with random miniBatch and evaluate testData
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
  const double updateRuleFactor = eta / miniBatch.size();

  // set nabla for biases and weights with all zeros
  vectorV nablaBiases;
  vectorM nablaWeights;
  for(vector<int>::iterator it = sizes.begin() + 1; it != sizes.end(); ++it) {
    int numNeurons = *it;
    int numNeuronsPreviewsLayer = *(it - 1);

    nablaBiases.push_back( gsl_vector_calloc(numNeurons) );
    nablaWeights.push_back( gsl_matrix_calloc(numNeurons, numNeuronsPreviewsLayer) );
  }
  
  // iterate over training pairs and build gradient descent nablas
  for (pair<gsl_vector*, int> trainingPair: miniBatch) {
    pair<vectorV, vectorM> deltaNablas = backprop(trainingPair);

    // add deltaNablaBiases from backprob(trainingsPair) to nablaBiases
    vectorV::const_iterator itNablaBiases = nablaBiases.begin();
    for(gsl_vector* deltaNablaBias: deltaNablas.first) { 
      gsl_vector_add(*itNablaBiases, deltaNablaBias);
      ++itNablaBiases;
    }

    // add deltaNablaWeights from backprob(trainingsPair) to nablaWeights
    vectorM::const_iterator itNablaWeights = nablaWeights.begin();
    for(gsl_matrix* deltaNablaWeight: deltaNablas.second) { 
      gsl_matrix_add(*itNablaWeights, deltaNablaWeight);
      ++itNablaWeights;
    }
  }

  // update weights via gradient descent update rule
  vectorV::const_iterator itBiases = biases.begin();
  for(gsl_vector* nablaBias: nablaBiases) {
    gsl_vector_scale(nablaBias, updateRuleFactor);
    gsl_vector_add(*itBiases, nablaBias);
    ++itBiases;
  }

  // update weights via gradient descent update rule
  vectorM::const_iterator itWeights = weights.begin();
  for(gsl_matrix* nablaWeight: nablaWeights) {
    gsl_matrix_scale(nablaWeight, updateRuleFactor);
    gsl_matrix_add(*itWeights, nablaWeight); 
    ++itWeights;
  }
}

//
// Back propagation
//
pair<vectorV, vectorM> Network::backprop(const pair<gsl_vector*, int> trainingPair) {
  // set zero biases and weights as placeholder
  vectorV nablaBiases;
  vectorM nablaWeights;
  for(vector<int>::iterator it = sizes.begin() + 1; it != sizes.end(); ++it) {
    int numNeurons = *it;
    int numNeuronsPreviewsLayer = *(it - 1);

    nablaBiases.push_back( gsl_vector_calloc (numNeurons) );
    nablaWeights.push_back( gsl_matrix_calloc(numNeurons, numNeuronsPreviewsLayer) );
  }

  gsl_vector* activation = trainingPair.first;
  vectorV activations;
  vectorV zVectors;
  activations.push_back(activation);

  //
  vectorM::const_iterator itWeights = weights.begin();
  for(gsl_vector* bias: biases) {
    gsl_vector* z = MatrixVectorMultiAndSum(*itWeights, activation, bias);
    zVectors.push_back(z);
    
    activation = gsl_vector_alloc(z->size);
    gsl_vector_memcpy(activation, z);
    SigmoidVectorized(activation);
    activations.push_back(activation);

    ++itWeights;
  }
  PrintVector(zVectors.last);

  return pair<vectorV, vectorM>(nablaBiases, nablaWeights);
}


