#include "camera_floor.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <stdexcept>
namespace B3Admission {
Decision Select(std::vector<Entry>& entries,size_t cap) {
    Decision d;
    // Read-only native counterfactual: identical previous/raw-count/ID order
    // and camera0-then-camera1 alternating selection, including short pools.
    std::array<std::vector<size_t>,2> camera;
    for(size_t i=0;i<entries.size();++i)camera.at(entries[i].core.id.first).push_back(i);
    for(auto& list:camera)std::sort(list.begin(),list.end(),[&](size_t ia,size_t ib){
        const auto& a=entries[ia];const auto& b=entries[ib];
        if(a.previous!=b.previous)return a.previous;
        return a.raw_observations!=b.raw_observations?a.raw_observations>b.raw_observations:a.core.id<b.core.id;
    });
    size_t rank=0;
    for(size_t i=0;;++i){bool any=false;for(const auto& list:camera)if(i<list.size()){
        auto& e=entries[list[i]];e.native_rank=rank;e.native_selected=rank++<cap;
        d.native[e.core.id.first]+=e.native_selected;any=true;
    }if(!any)break;}
    std::map<HiltiLandmarkId,Entry> metadata;
    std::vector<CAdmission::Candidate> cs;
    for(const auto& e:entries){metadata.emplace(e.core.id,e);cs.push_back(e.core);}
    // Apply global quality ranking with all prior variables protected.
    CAdmission::Select(cs,cap);
    entries.clear();
    for(const auto& c:cs){auto e=metadata.at(c.id);e.core=c;e.B_selected=c.selected;
        d.B[c.id.first]+=c.selected;entries.push_back(e);}
    d.final=d.B;d.activated=d.B[1]<d.native[1];
    if(!d.activated)return d; // exact original B IDs, ordering and factors
    size_t replacement=0;
    while(d.final[1]<d.native[1]){
        while(replacement<entries.size() && (entries[replacement].core.id.first!=1 || entries[replacement].core.selected))++replacement;
        if(replacement==entries.size())break;
        size_t donor=entries.size();
        while(donor>0){--donor;const auto& c=entries[donor].core;
            if(c.id.first==0 && c.selected && !c.mandatory)break;}
        const auto& c=entries[donor].core;
        if(c.id.first!=0 || !c.selected || c.mandatory){d.blocked_by_prior=true;break;}
        d.swaps.emplace_back(c.id,entries[replacement].core.id);
        entries[donor].core.selected=false;entries[replacement].core.selected=true;
        --d.final[0];++d.final[1];++replacement;
    }
    for(const auto& e:entries)if(e.core.mandatory&&!e.core.selected)throw std::runtime_error("B3 lost protected Admission-B prior ID");
    return d;
}
}
