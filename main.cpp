#include "Network.h"
#include "misc.h"

using namespace std;

int main( int argc, const char* argv[] )
{
  cout << "Network\n";
  gsl_vector* activation = gsl_vector_alloc(2);

  for (int i=0; i<activation->size; i++){
    gsl_vector_set(activation, i, i);
  }

  Network Network({2, 3, 1});
  cout << endl;
  
  DataSet trainingData = DataSet(500);
  DataSet testData = DataSet(100); 
  Network.SGD(trainingData, 30, 10, 3.0, testData);
  PrintVector(Network.feedforward(activation));
  cout << endl;
  
  return 0;
}