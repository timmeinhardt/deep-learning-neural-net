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

	// set random biases vectors and weights matrices for each layer 
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
  gsl_vector* a = CopyOfGslVector(activation);
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
  const double updateRuleFactor = eta / miniBatchSize;
  const int nTestData = testData.size();

  // iterate over training epochs, train network with random miniBatch and evaluate testData
  for (int i=1; i<=epochs; i++){
    random_shuffle( trainingData.begin(), trainingData.end() );

    // split trainingData and train with miniBatch
    for (DataSet::iterator startMiniBatch = trainingData.begin(); startMiniBatch != trainingData.end(); startMiniBatch = startMiniBatch + miniBatchSize) {
      // build miniBatch from trainingsData
      DataSet::iterator endMiniBatch = startMiniBatch + miniBatchSize;
      DataSet miniBatch(startMiniBatch, endMiniBatch);

      trainWithMiniBatch(miniBatch, updateRuleFactor);
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
void Network::trainWithMiniBatch(const DataSet& miniBatch, const double& updateRuleFactor) {
  // set nabla for biases and weights with all zeros
  pair<vectorV, vectorM> placeholders = mallocPlaceholders();
  vectorV & nablaBiases   = placeholders.first;
  vectorM & nablaWeights  = placeholders.second;
  
  // iterate over training pairs and build gradient descent nablas
  for (pair<gsl_vector*, int> trainingPair: miniBatch) {
    pair<vectorV, vectorM> deltaNablas = backprop(trainingPair);

    // add deltaNablaBiases from backprob(trainingPair) to nablaBiases
    vectorV::const_iterator itNablaBiases = nablaBiases.begin();
    for(gsl_vector* deltaNablaBias: deltaNablas.first) { 
      gsl_vector_add(*itNablaBiases, deltaNablaBias);
      ++itNablaBiases;
    }

    // add deltaNablaWeights from backprob(trainingPair) to nablaWeights
    vectorM::const_iterator itNablaWeights = nablaWeights.begin();
    for(gsl_matrix* deltaNablaWeight: deltaNablas.second) { 
      gsl_matrix_add(*itNablaWeights, deltaNablaWeight);
      ++itNablaWeights;
    }
    
    freePlaceholders(deltaNablas);
  }

  // update weights via gradient descent update rule
  vectorV::const_iterator nablaBias = nablaBiases.begin();
  for(gsl_vector* bias: biases) {
    gsl_vector_add_constant(bias, - updateRuleFactor);
    gsl_vector_mul(bias, *nablaBias);
    ++nablaBias;
  }

  // update weights via gradient descent update rule
  vectorM::const_iterator nablaWeight = nablaWeights.begin();
  for(gsl_matrix* weight: weights) {
    gsl_matrix_add_constant(weight, - updateRuleFactor);
    gsl_matrix_mul(weight, *nablaWeight); 
    ++nablaWeight;
  }

  freePlaceholders(placeholders);
}

//
// Back propagation
//
pair<vectorV, vectorM> Network::backprop(const pair<gsl_vector*, int> trainingPair) {
  // set zero biases and weights as placeholder
  pair<vectorV, vectorM> nablas = mallocPlaceholders();
  vectorV& nablaBiases   = nablas.first;
  vectorM& nablaWeights  = nablas.second;

  // set training output vector from trainingPair
  gsl_vector* output = gsl_vector_calloc(sizes.back());
  gsl_vector_set(output, trainingPair.second, 1.0);

  gsl_vector* activation = trainingPair.first;
  vectorV activations;
  vectorV zVectors;
  activations.push_back(activation);

  // fill activation and z vectors
  vectorM::const_iterator itWeights = weights.begin();
  for(gsl_vector* bias: biases) {
    gsl_vector* z = MatrixVectorMultiAndSum(*itWeights, activation, bias);
    zVectors.push_back(z);
   
    // allocate new activation vector so the z vector is not changed 
    activation = CopyOfGslVector(z);
    SigmoidVectorized(activation);
    activations.push_back(activation);

    ++itWeights;
  }

  // delta = (activations.back() - output) * zVectors.back()
  gsl_vector* delta = CopyOfGslVector(activations.back());
  gsl_vector_sub(delta, output);
  gsl_vector* spv = SigmoidPrimeVectorized(zVectors.end()[-1]);
  gsl_vector_mul(delta, spv);

  nablaBiases.end()[-1] = delta;
  gsl_matrix_mul_for_vectors(nablaWeights.end()[-1], delta, activations.end()[-2]);

  for (int l = 2; l < numLayers; l++) {
    spv   = SigmoidPrimeVectorized(zVectors.end()[-l]);

    // delta = (weights[-l+1] * delta) * spv
    delta = gsl_vector_alloc(( weights.end()[-l+1] )->size2);
    gsl_blas_dgemv(CblasTrans, 1.0, weights.end()[-l+1], nablaBiases.end()[-l+1], 0.0, delta); 
    gsl_vector_mul(delta, spv);

    nablaBiases.end()[-l]  = delta;
    gsl_matrix_mul_for_vectors(nablaWeights.end()[-l], delta, activations.end()[-l-1]);
  }

  return nablas;
}

//
// Returns pair of all zero biases and weights
//
pair<vectorV, vectorM> Network::mallocPlaceholders() {
  vectorV placeholderBiases;
  vectorM placeholderWeights;
  for(vector<int>::iterator it = sizes.begin() + 1; it != sizes.end(); ++it) {
    int numNeurons = *it;
    int numNeuronsPreviewsLayer = *(it - 1);

    placeholderBiases.push_back( gsl_vector_calloc (numNeurons) );
    placeholderWeights.push_back( gsl_matrix_calloc(numNeurons, numNeuronsPreviewsLayer) );
  }

  return pair<vectorV, vectorM>(placeholderBiases, placeholderWeights);
}

//
// Free placeholder vectors
//
void Network::freePlaceholders(pair<vectorV, vectorM> placeholders) {
  for(gsl_vector* bias: placeholders.first) {
    gsl_vector_free(bias);
  }

  for(gsl_matrix* weight: placeholders.second) {
    gsl_matrix_free(weight);
  }
}

