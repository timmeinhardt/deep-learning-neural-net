#include "Network.h"
#include "misc.h"

#include "mnist_loader.cpp"

using namespace std;

int main( int argc, const char* argv[] )
{
  cout << "Network" << endl;

  int numNeuronsInput = 748;
  Network Network({numNeuronsInput, 30, 10});
  
  DataSet trainingData = BuildDataSet(500, numNeuronsInput);
  DataSet testData = BuildDataSet(100, numNeuronsInput);

  Network.SGD(trainingData, 10, 10, 3.0, testData);

  return 0;
}