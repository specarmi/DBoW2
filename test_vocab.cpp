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

void loadFeatures(vector<vector<cv::Mat > > &features);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 5;

// ----------------------------------------------------------------------------

int main()
{
  // load the vocabulary from disk
  SuperPointVocabulary voc("superpt_voc.yml.gz");

  // get features for a small number of images
  vector<vector<cv::Mat > > features;
  loadFeatures(features);

  // score images to test vocab
  cout << "Scoring image matches (0 low, 1 high): " << endl;
  cout << "Images from FLIR ADAS videos frames: 1, 2, 15, 40, 4224" << endl;
  BowVector v1, v2;
  for(int i = 0; i < NIMAGES; i++)
  {
    voc.transform(features[i], v1);
    for(int j = 0; j < NIMAGES; j++)
    {
      voc.transform(features[j], v2);
      
      double score = voc.score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<cv::Mat > > &features)
{
  features.clear();
  features.reserve(NIMAGES);

  cout << "Importing SuperPoint features..." << endl;
  for(int i = 1; i < NIMAGES + 1; ++i)
  {
    stringstream ss;
    ss << "../Selection_Keypts_and_Desc/" << i << ".yaml";
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