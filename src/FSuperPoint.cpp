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
      mean += *descriptors[i];
    }
    mean = mean / (float)descriptors.size();
  }
}

// --------------------------------------------------------------------------
  
double FSuperPoint::distance(const FSuperPoint::TDescriptor &a, 
  const FSuperPoint::TDescriptor &b)
{
  double sqd = 0.0;
  const float *a_ptr=a.ptr<float>(0);
  const float *b_ptr=b.ptr<float>(0);
  for(int i = 0; i < FSuperPoint::L; i ++)
      sqd += (a_ptr[i] - b_ptr[i])*(a_ptr[i] - b_ptr[i]);
  return sqd;
}

// --------------------------------------------------------------------------
  
std::string FSuperPoint::toString(const FSuperPoint::TDescriptor &a)
{
  stringstream ss;
  const float *p = a.ptr<float>();
  
  for(int i = 0; i < FSuperPoint::L; ++i, ++p)
  {
    ss << (float)*p << " ";
  }
  
  return ss.str();
}

// --------------------------------------------------------------------------
  
void FSuperPoint::fromString(FSuperPoint::TDescriptor &a, const std::string &s)
{
  a.create(1, FSuperPoint::L, CV_32F);
  float *p = a.ptr<float>();
  
  stringstream ss(s);
  for(int i = 0; i < FSuperPoint::L; ++i, ++p)
  {
    float f;
    ss >> f;
    
    if(!ss.fail()) 
      *p = (float)f;
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
    descriptors[i].row(0).copyTo(mat.row(i));
  } 
}

// --------------------------------------------------------------------------

} // namespace DBoW2

