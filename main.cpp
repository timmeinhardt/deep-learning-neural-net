#include "Network.h"
#include "misc.h"

#include "loader.cpp"

using namespace std;

int main( int argc, const char* argv[] )
{
  cout << "Network" << endl;

  const char* imageFile = "train-images-idx3-ubyte";
  const char* labelFile = "train-labels-idx1-ubyte";

  Loader Loader;
  DataSet mnistData;
  cout << "load data..." << endl;
  Loader.Parse(mnistData, imageFile, labelFile);

  DataSet trainingData(mnistData.begin() + 1000, mnistData.begin() + 5000);
  DataSet testData(mnistData.begin(), mnistData.begin() + 1000 );

  int numNeuronsInput = Loader.GetImageSize();
  Network Network({numNeuronsInput, 30, 10});
  cout << "train network..." << endl;
  Network.SGD(trainingData, 10, 10, 3.0, testData);

  return 0;
}