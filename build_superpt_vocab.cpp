#include <iostream>
#include <vector>

// DBoW2
#include "DBoW2.h" // defines SuperPointVocabulary

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/persistence.hpp>


using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(vector<vector<cv::Mat > > &features, const string &feature_filepath);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testVocCreation(const vector<vector<cv::Mat > > &features);
void testDatabase(const vector<vector<cv::Mat > > &features);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 8862;

// ----------------------------------------------------------------------------

int main(int argc, char **argv)
{
  if(argc != 2)
  {
      cerr << endl << "Usage: ./build_superpt_vocab path_to_superpoint_features" << endl;
      return 1;
  }

  vector<vector<cv::Mat > > features;
  loadFeatures(features, argv[1]);

  testVocCreation(features);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<cv::Mat > > &features, const string &feature_filepath)
{
  features.clear();
  features.reserve(NIMAGES);

  cout << "Importing SuperPoint features..." << endl;
  for(int i = 1; i < NIMAGES + 1; ++i)
  {
    stringstream ss;
    ss << feature_filepath << i << ".yaml";
    cv::FileStorage pts_log(ss.str(), cv::FileStorage::READ);

    cv::Mat descriptors;
    pts_log["descriptors"] >> descriptors;

    features.push_back(vector<cv::Mat >());
    changeStructure(descriptors, features.back());
  }
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<cv::Mat > > &features)
{
  // branching factor and depth levels 
  const int k = 10;
  const int L = 5;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L1_NORM;

  SuperPointVocabulary voc(k, L, weight, scoring);

  cout << "Creating " << k << "^" << L << " SuperPoint vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("superpt_voc.yml.gz");
  cout << "Done" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;
}
