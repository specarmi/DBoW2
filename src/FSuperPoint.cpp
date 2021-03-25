#include <vector>
#include <string>
#include <sstream>

#include "FSuperPoint.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

void FSuperPoint::meanValue(const std::vector<FSuperPoint::pDescriptor> &descriptors, 
  FSuperPoint::TDescriptor &mean)
{
  if(descriptors.empty())
  {
    mean.release();
    return;
  }
  else if(descriptors.size() == 1)
  {
    mean = descriptors[0]->clone();
  }
  else
  {
    mean = cv::Mat::zeros(1, FSuperPoint::L, CV_32F);
    for(size_t i = 0; i < descriptors.size(); ++i)
    {
      mean = mean + *descriptors[i];
    }
    mean = mean / (float)descriptors.size();
  }
}

// --------------------------------------------------------------------------
  
double FSuperPoint::distance(const FSuperPoint::TDescriptor &a, 
  const FSuperPoint::TDescriptor &b)
{
  return cv::norm(a, b);
}

// --------------------------------------------------------------------------
  
std::string FSuperPoint::toString(const FSuperPoint::TDescriptor &a)
{
  stringstream ss;
  const double *p = a.ptr<double>();
  
  for(int i = 0; i < a.cols; ++i, ++p)
  {
    ss << (double)*p << " ";
  }
  
  return ss.str();
}

// --------------------------------------------------------------------------
  
void FSuperPoint::fromString(FSuperPoint::TDescriptor &a, const std::string &s)
{
  a.create(1, FSuperPoint::L, CV_32F);
  double *p = a.ptr<double>();
  
  stringstream ss(s);
  for(int i = 0; i < FSuperPoint::L; ++i, ++p)
  {
    double d;
    ss >> d;
    
    if(!ss.fail()) 
      *p = (double)d;
  }
}

// --------------------------------------------------------------------------

void FSuperPoint::toMat32F(const std::vector<TDescriptor> &descriptors, 
  cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }
  
  const int N = descriptors.size();
  
  mat.create(N, FSuperPoint::L, CV_32F);
  
  for(int i = 0; i < N; ++i)
  {
    const double *desc = descriptors[i].ptr<double>();
    float *p = mat.ptr<float>(i);
    for(int j = 0; j < FSuperPoint::L; ++j, ++p)
    {
      *p = (float)desc[j];
    }
  } 
}

// --------------------------------------------------------------------------

} // namespace DBoW2

