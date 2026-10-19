#pragma once
#include "quality.h"
#include <array>
namespace B3Admission {
using namespace myslam;
struct Entry {
    CAdmission::Candidate core;
    bool previous=false,native_selected=false,B_selected=false;
    size_t raw_observations=0,native_rank=0;
};
struct Decision {
    std::array<size_t,2> native{{}},B{{}},final{{}};
    bool activated=false,blocked_by_prior=false;
    std::vector<std::pair<HiltiLandmarkId,HiltiLandmarkId>> swaps;
};
Decision Select(std::vector<Entry>&,size_t);
}
