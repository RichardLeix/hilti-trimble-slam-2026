#include "myslam/hilti_local_map.h"
#include "myslam/hilti_camera_information.h"
#include "myslam/hilti_rolling_shutter.h"
#include "myslam/hilti_landmark_quarantine.h"
#include "synthetic_window.h"
#include <gtest/gtest.h>
#include <fstream>
#include <iomanip>
#include <set>
using namespace myslam;
TEST(H2,OnlyRetainedWindowOptionChanges){
 const HiltiBackendOptions a;
 auto b=LoadHiltiBackendOptions(std::string(H2_SOURCE_DIR)+"/config/hilti_backend.yaml");
 EXPECT_EQ(a.window_size,7);EXPECT_EQ(b.window_size,11);
 EXPECT_TRUE(b.experimental_landmark_quarantine_B);
#define SAME(field) EXPECT_EQ(a.field,b.field)
 SAME(center_row_rolling_shutter);SAME(enable_marginalization);SAME(landmark_quarantine_log_path);SAME(marginalization_rank_tolerance);SAME(keyframe_stride);SAME(max_landmarks);SAME(min_landmarks);SAME(max_iterations);SAME(bearing_sigma_rad);SAME(visual_huber_sigma);SAME(max_initial_bearing_rad);SAME(max_landmark_bearing_rad);SAME(velocity_prior_sigma);SAME(gyro_prior_sigma);SAME(accel_prior_sigma);SAME(max_pose_update_m);SAME(max_rotation_update_rad);SAME(max_velocity_update);SAME(max_gyro_update);SAME(max_accel_update);
#undef SAME
}
static void Extend(SyntheticWindow& d){
 const auto initial=d.truth.keyframes.front().state;d.truth.keyframes.clear();d.truth.intervals.clear();
 for(int i=0;i<13;++i){HiltiKeyframeState f;f.timestamp_ns=kEpoch+i*75000000LL;f.exposure_timestamps_ns={{f.timestamp_ns,f.timestamp_ns+20000}};
  if(i){auto delta=PreintegrateImu(d.samples,d.truth.keyframes.back().timestamp_ns,f.timestamp_ns,d.truth.calibration,initial.bias);d.truth.intervals.push_back(delta);f.state=PropagateNavState(d.truth.keyframes.back().state,delta,d.truth.gravity_w);}else f.state=initial;
  f.exposure_deltas[1]=PreintegrateImu(d.samples,f.timestamp_ns,f.timestamp_ns+20000,d.truth.calibration,initial.bias);d.truth.keyframes.push_back(f);}
 for(auto& l:d.truth.landmarks){l.observations.clear();for(const auto& f:d.truth.keyframes){const Vec3 p=HiltiCameraWorldPose(f,l.id.first,d.truth.cameras,d.truth.gravity_w).inverse()*l.position_w;Vec2 pixel;ASSERT_TRUE(d.truth.cameras[l.id.first]->Project(p,pixel));Vec3 ray;ASSERT_TRUE(d.truth.cameras[l.id.first]->Unproject(pixel,ray));HiltiBearingObservation o;o.timestamp_ns=f.timestamp_ns;o.bearing=ray;l.observations.push_back(o);}}
}
static void CheckJob(const HiltiBackendJob& j){
 ASSERT_EQ(j.intervals.size()+1,j.keyframes.size());std::set<int64_t> ts;std::set<HiltiLandmarkId> ids;
 for(size_t i=0;i<j.keyframes.size();++i){const auto t=j.keyframes[i].timestamp_ns;EXPECT_TRUE(ts.insert(t).second);EXPECT_EQ(j.keyframes[i].exposure_timestamps_ns[0],t);if(i){EXPECT_LT(j.keyframes[i-1].timestamp_ns,t);EXPECT_EQ(j.intervals[i-1].begin_timestamp_ns,j.keyframes[i-1].timestamp_ns);EXPECT_EQ(j.intervals[i-1].end_timestamp_ns,t);EXPECT_NEAR(j.intervals[i-1].dt,(t-j.keyframes[i-1].timestamp_ns)*1e-9,1e-12);}}
 for(const auto& l:j.landmarks){EXPECT_TRUE(ids.insert(l.id).second);EXPECT_GE(l.id.first,0);EXPECT_LE(l.id.first,1);EXPECT_LT(l.id.second,50);int64_t prev=0;for(const auto& o:l.observations){EXPECT_TRUE(ts.count(o.timestamp_ns));EXPECT_GT(o.timestamp_ns,prev);prev=o.timestamp_ns;}}
}
TEST(H2,TwelveSourceElevenRetainedFIFOAndConnectedPrior){
 SyntheticWindow d;Extend(d);d.options.window_size=11;d.options.max_landmarks=40;d.options.enable_marginalization=true;d.options.experimental_landmark_quarantine_B=true;
 HiltiLocalMap map(d.truth.cameras,d.truth.calibration,d.options,d.truth.gravity_w);for(const auto& l:d.truth.landmarks)map.Seed(l.id,l.position_w);
 HiltiLandmarkQuarantine quarantine;uint64_t version=0;
 for(size_t i=0;i<d.truth.keyframes.size();++i){const auto& f=d.truth.keyframes[i];map.AddKeyframe(f.timestamp_ns,f.state,f.exposure_timestamps_ns,d.Tracks(i),d.samples);ASSERT_EQ(map.WindowSize(),std::min(size_t(11),i+1));EXPECT_EQ(map.Window().front().timestamp_ns,d.truth.keyframes[i<11?0:i-10].timestamp_ns);if(i<10)continue;
  ++version;auto job=map.Snapshot(version,version-1);CheckJob(job);ASSERT_EQ(job.keyframes.size(),11);EXPECT_EQ(job.keyframes.back().timestamp_ns,f.timestamp_ns);
  if(i>10){ASSERT_TRUE(job.marginalization_source);const auto& source=*job.marginalization_source;CheckJob(source);ASSERT_EQ(source.keyframes.size(),12);EXPECT_EQ(source.keyframes.front().timestamp_ns,d.truth.keyframes[i-11].timestamp_ns);EXPECT_EQ(job.keyframes.front().timestamp_ns,source.keyframes[1].timestamp_ns);for(size_t k=0;k<11;++k)EXPECT_EQ(job.keyframes[k].timestamp_ns,source.keyframes[k+1].timestamp_ns);}
  auto result=OptimizeHiltiWindow(job,d.options,&quarantine);ASSERT_TRUE(result.diagnostics.accepted)<<result.diagnostics.reason;ASSERT_TRUE(result.factor_snapshot);EXPECT_EQ(result.keyframes.size(),11);
  
  if(i>10){ASSERT_TRUE(result.prior);int dimension=0;std::set<MarginalizationKey> keys;std::set<int64_t> states;std::set<HiltiLandmarkId> points;for(const auto& k:job.keyframes)states.insert(k.timestamp_ns);for(const auto& l:job.landmarks)points.insert(l.id);for(const auto& v:result.prior->linearization){dimension+=v.Dimension();EXPECT_TRUE(keys.insert(v.key).second);if(v.key.kind==MarginalizationKey::Kind::State)EXPECT_TRUE(states.count(v.key.timestamp_ns));else EXPECT_TRUE(points.count({v.key.camera_id,v.key.feature_id}));}EXPECT_EQ(result.prior->hessian.rows(),dimension);EXPECT_EQ(result.prior->hessian.cols(),dimension);EXPECT_EQ(result.prior->gradient.size(),dimension);EXPECT_TRUE(result.prior->hessian.allFinite());EXPECT_TRUE(result.prior->gradient.allFinite());EXPECT_GT(result.prior->rank,0);}
  map.Apply(result,{});
 }
}
TEST(H2,NativeTimestampAndDuplicateProtectionsRemain){
 SyntheticWindow d;d.options.window_size=11;HiltiLocalMap map(d.truth.cameras,d.truth.calibration,d.options,d.truth.gravity_w);const auto& f=d.truth.keyframes[0];map.AddKeyframe(f.timestamp_ns,f.state,f.exposure_timestamps_ns,d.Tracks(0),d.samples);EXPECT_THROW(map.AddKeyframe(f.timestamp_ns,f.state,f.exposure_timestamps_ns,d.Tracks(0),d.samples),std::invalid_argument);
 auto bad=d.samples;bad[10].angular_velocity[0]+=1;const auto& next=d.truth.keyframes[1];EXPECT_THROW(map.AddKeyframe(next.timestamp_ns,next.state,next.exposure_timestamps_ns,d.Tracks(1),bad),std::invalid_argument);
}

TEST(H2,FixedCameraInformationAndActiveBudget) {
 const auto options=LoadHiltiBackendOptions(std::string(H2_SOURCE_DIR)+"/config/hilti_backend.yaml");
 EXPECT_DOUBLE_EQ(HiltiOfflineCameraInformation(0),1.0);
 EXPECT_DOUBLE_EQ(HiltiOfflineCameraInformation(1),0.5);
 EXPECT_EQ(options.max_landmarks,240);
 EXPECT_EQ(options.keyframe_stride,5);
 EXPECT_TRUE(options.enable_marginalization);
 EXPECT_TRUE(options.experimental_landmark_quarantine_B);
 EXPECT_EQ(options.window_size,11);
 EXPECT_NEAR(10.0*options.keyframe_stride/29.97,1.668333,0.00001);
}
