#include "camera_floor.h"
#include "synthetic_window.h"
#include <gtest/gtest.h>
#include <random>
#include <set>
using namespace B3Admission;
static Entry Make(int camera,uint64_t id,double q,bool mandatory=false){
    Entry e;e.core.id={camera,id};e.core.geometry.observations=4;e.core.geometry.parallax_sigma=20;
    e.core.geometry.relative_depth_sigma=.05;e.core.geometry.residual_sigma=1;e.core.quality=q;
    e.core.visible=true;e.core.mandatory=mandatory;e.raw_observations=4;return e;
}
TEST(B3,GlobalCam1GrowthPassesThroughExactBWithoutReordering){
    std::vector<Entry> es;for(int k=0;k<2;++k)for(int i=0;i<8;++i)es.push_back(Make(k,i,k?.99:.3));
    auto d=Select(es,8);EXPECT_FALSE(d.activated);EXPECT_EQ(d.B,d.final);EXPECT_EQ(d.final[1],8);
    std::vector<CAdmission::Candidate> cs;for(const auto& e:es)cs.push_back(e.core);CAdmission::Select(cs,8);
    for(size_t i=0;i<cs.size();++i){EXPECT_EQ(cs[i].id,es[i].core.id);EXPECT_EQ(cs[i].selected,es[i].core.selected);}
}
TEST(B3,OnlyLowestOptionalCam0SwappedForBestRejectedCam1){
    std::vector<Entry> es;for(int i=0;i<8;++i){es.push_back(Make(0,i,.95-.01*i,i==7));es.push_back(Make(1,i,.5-.01*i));}
    auto d=Select(es,8);EXPECT_TRUE(d.activated);EXPECT_EQ(d.native[1],4);EXPECT_EQ(d.B[1],0);EXPECT_EQ(d.final[1],4);ASSERT_EQ(d.swaps.size(),4);
    EXPECT_EQ(d.swaps[0].first,HiltiLandmarkId(0,6));EXPECT_EQ(d.swaps[0].second,HiltiLandmarkId(1,0));
    EXPECT_TRUE(es.front().core.mandatory);for(const auto& e:es)if(e.core.mandatory)EXPECT_TRUE(e.core.selected);
}
TEST(B3,NativeFloorHandlesShortPoolsAndOddCapsWithoutFixedQuota){
    std::vector<Entry> es{Make(0,0,.99)};for(int i=0;i<9;++i)es.push_back(Make(1,i,.4));
    auto d=Select(es,7);EXPECT_EQ(d.native[0],1);EXPECT_EQ(d.native[1],6);EXPECT_EQ(d.final[1],6);EXPECT_FALSE(d.activated);
    es.clear();for(int k=0;k<2;++k)for(int i=0;i<8;++i)es.push_back(Make(k,i,k?.4:.99));
    d=Select(es,7);EXPECT_EQ(d.native[0],4);EXPECT_EQ(d.native[1],3);EXPECT_EQ(d.final[0],4);EXPECT_EQ(d.final[1],3);
}
TEST(B3,PriorProtectionDominatesInfeasibleFloorWithoutCapChange){
    std::vector<Entry> es;for(int i=0;i<6;++i)es.push_back(Make(0,i,.9,true));for(int i=0;i<8;++i)es.push_back(Make(1,i,.3));
    auto d=Select(es,8);EXPECT_TRUE(d.activated);EXPECT_TRUE(d.blocked_by_prior);EXPECT_EQ(d.final[1],2);EXPECT_EQ(d.native[1],4);
    size_t count=0;for(const auto& e:es){count+=e.core.selected;if(e.core.mandatory)EXPECT_TRUE(e.core.selected);}EXPECT_EQ(count,8);
}
TEST(B3,RandomizedFloorAndRankingInvariants){
    std::mt19937 rng(71015);
    for(int trial=0;trial<300;++trial){std::vector<Entry> es;size_t cap=1+rng()%40;
        for(int k=0;k<2;++k)for(size_t i=0,n=rng()%60;i<n;++i){auto e=Make(k,i,(rng()%1000)/1000.);e.previous=rng()%2;e.raw_observations=1+rng()%12;es.push_back(e);}
        auto d=Select(es,cap);size_t selected=0,cam1=0;
        for(const auto& e:es){selected+=e.core.selected;cam1+=e.core.selected&&e.core.id.first==1;}
        EXPECT_EQ(selected,std::min(cap,es.size()));EXPECT_EQ(cam1,std::max(d.B[1],d.native[1]));EXPECT_FALSE(d.blocked_by_prior);
        if(!d.activated)for(const auto& e:es)EXPECT_EQ(e.core.selected,e.B_selected);
        for(size_t i=1;i<es.size();++i)EXPECT_FALSE(CAdmission::Better(es[i].core,es[i-1].core));
    }
}
static std::unique_ptr<HiltiLocalMap> MakeLocalMap(SyntheticWindow& d,HiltiBackendOptions o){
    std::unique_ptr<HiltiLocalMap> m(new HiltiLocalMap(d.truth.cameras,d.truth.calibration,o,d.truth.gravity_w));
    for(const auto& l:d.truth.landmarks)m->Seed(l.id,l.position_w);
    for(size_t i=0;i<d.truth.keyframes.size();++i){const auto& f=d.truth.keyframes[i];m->AddKeyframe(f.timestamp_ns,f.state,f.exposure_timestamps_ns,d.Tracks(i),d.samples);}return m;
}
TEST(B3,SourcePriorAndAllEligibleBProtectedLandmarksRemainIntact){
    SyntheticWindow d;d.options.enable_marginalization=true;d.options.max_landmarks=60;auto map=MakeLocalMap(d,d.options);
    auto old=d.truth;old.landmarks.resize(40);auto prior=std::make_shared<MarginalizationPrior>();
    for(const auto& l:old.landmarks){MarginalizationVariable v;v.key=MarginalizationKey::Landmark(l.id.first,l.id.second);v.point_w=l.position_w;prior->linearization.push_back(v);}old.prior=prior;
    HiltiBackendResult correction;correction.factor_snapshot=std::make_shared<const HiltiBackendJob>(old);correction.landmarks=old.landmarks;correction.keyframes=old.keyframes;map->Apply(correction,{});
    const auto selected=map->Snapshot(2,1);
    ASSERT_EQ(selected.landmarks.size(),60);ASSERT_TRUE(selected.marginalization_source);EXPECT_EQ(selected.marginalization_source->prior.get(),prior.get());
    for(const auto& l:old.landmarks)EXPECT_TRUE(std::any_of(selected.landmarks.begin(),selected.landmarks.end(),[&](const HiltiLandmark& p){return p.id==l.id;}));
    EXPECT_GE(selected.marginalization_source->landmarks.size(),60);
}
