#pragma once
#include "myslam/hilti_local_map.h"
#include "myslam/hilti_landmark_weighting.h"
#include <vector>
namespace CAdmission {
using namespace myslam;
struct Candidate {
    size_t camera_index=0;
    HiltiLandmarkId id;
    bool mandatory=false,visible=false,selected=false;
    LandmarkWeightQuality geometry;
    double quality=0,age_s=0;
};
double Quality(const LandmarkWeightQuality&);
bool Better(const Candidate&,const Candidate&);
void Select(std::vector<Candidate>&,size_t);
HiltiLandmark ValidCopy(const HiltiBackendJob&,const HiltiLandmark&,const HiltiBackendOptions&);
}
