#include "myslam/frontend.h"
#include "myslam/hilti_initializer.h"
#include "myslam/global_trajectory.h"
#include "myslam/hilti_rolling_shutter.h"
#include <fstream>
#include <iomanip>
#include <map>
#include <stdexcept>

// Complete production replay. Accepts sensor/calibration inputs only; GT is not
// linked into this runner. Backend drain/capture timing matches the saved baseline.
namespace {
using namespace myslam;
void Vector(std::ostream& out,const Vec3& v) {out<<','<<v.x()<<','<<v.y()<<','<<v.z();}
void Pose(std::ostream& out,const SE3& pose) {
    Vector(out,pose.translation());const auto q=pose.unit_quaternion();out<<','<<q.x()<<','<<q.y()<<','<<q.z()<<','<<q.w();
}
void Quote(std::ostream& out,const std::string& text) {out<<'"';for(char c:text){if(c=='"')out<<"\"\"";else out<<c;}out<<'"';}
void State(std::ostream& out,const NavState& state,const Vec3& gravity) {
    Pose(out,state.Pose());Vector(out,state.velocity_w);Vector(out,state.bias.gyroscope);
    Vector(out,state.bias.accelerometer);Vector(out,gravity);
}
struct DenseRecord {
    size_t epoch = 0, frame_index = 0;
    int64_t timestamp_ns = 0;
    HiltiFrontendStatus status = HiltiFrontendStatus::Buffering;
    uint64_t backend_version = 0;
    Frame::Ptr frame;
};
void DenseRow(std::ostream& out,const DenseRecord& record,HiltiFrontendStatus status,
              const NavState* state,const Vec3& gravity) {
    out<<record.epoch<<','<<record.timestamp_ns<<','<<HiltiFrontendStatusName(status)<<','<<record.backend_version;
    if(state)State(out,*state,gravity);else for(int k=0;k<19;++k)out<<",nan";
    out<<'\n';
}
void Keyframes(const std::string& path,const HiltiAlignedVector<PoseGraphPose>& poses) {
    std::ofstream file(path);file<<std::setprecision(17)<<"id,timestamp_ns,px,py,pz,qx,qy,qz,qw\n";
    for(const auto& p:poses){file<<p.keyframe_id<<','<<p.timestamp_ns;Pose(file,p.body_pose);file<<'\n';}
    if(!file)throw std::runtime_error("cannot write keyframes");
}
}
int main(int argc,char** argv) {
    google::InitGoogleLogging(argv[0]);
    if(argc<9) {
        std::cerr<<"Usage: run_hilti_sequence <rosbag> <imu.yaml> <camera.yaml> <loop.yaml> <orb.fbow> <graph.yaml> <global.yaml> <output-prefix> --backend-config <yaml> --startup-config <yaml>\n";return 2;
    }
    try {
        HiltiFrontendOptions options;std::string startup_config,backend_config;
        HiltiBackendOptions backend_options;
        for(int i=9;i<argc;++i) {
            const std::string option=argv[i];
            if(option=="--backend-config" && i+1<argc){backend_config=argv[++i];backend_options=LoadHiltiBackendOptions(backend_config);}
            else if(option=="--startup-config" && i+1<argc)startup_config=argv[++i];
            else throw std::invalid_argument("unknown or incomplete Hilti runtime option");
        }
        if(backend_config.empty() || startup_config.empty())
            throw std::invalid_argument("H2 requires explicit backend and startup configurations");
        if(backend_options.window_size!=11 || !backend_options.experimental_landmark_quarantine_B)
            throw std::invalid_argument("H2 requires retained states 11 and landmark quarantine");
        options=LoadHiltiFrontendOptions(startup_config);
        options.diagnostic_legacy_startup=false;
        std::cout<<"Hilti backend: center-row rolling shutter; physical timestamp reference unverified; readout "
            <<std::setprecision(17)<<HiltiReadoutSeconds<<" s, top to bottom\n";
        const std::string prefix=argv[8];
        if(backend_options.experimental_landmark_quarantine_B) {
            backend_options.landmark_quarantine_log_path=prefix+"_landmark_quarantine.csv";
            std::cout<<"H2 landmark quarantine (active/source/prior; fixed gates)\n";
        }
        const auto calibration=LoadImuCalibration(argv[2],argv[3]);HiltiDataset dataset(argv[1]);
        if(!dataset.Init()||!dataset.LoadCameraCalibration(argv[3]))throw std::runtime_error(dataset.LastError());
        HiltiCameras cameras;for(int k=0;k<2;++k)cameras[k]=std::dynamic_pointer_cast<const EucmCamera>(dataset.GetCamera(k));
        auto backend=std::make_shared<HiltiBackend>(backend_options);auto loops=std::make_shared<LoopClosing>(cameras,LoadLoopClosingOptions(argv[4],argv[5]));
        auto global=std::make_shared<GlobalTrajectory>(loops,LoadPoseGraphOptions(argv[6]),LoadGlobalTrajectoryOptions(argv[7]));
        Frontend frontend(cameras,calibration,options);frontend.SetHiltiBackend(backend);frontend.SetHiltiLoopClosing(loops);frontend.SetHiltiGlobalTrajectory(global);
        std::ofstream local(prefix+"_local.csv"),epochs(prefix+"_telemetry_epochs.csv"),windows(prefix+"_telemetry_windows.csv");
        if(!local||!epochs||!windows)throw std::runtime_error("cannot create runtime artifacts");
        local<<std::setprecision(17)<<"epoch,timestamp_ns,status,backend_version,px,py,pz,qx,qy,qz,qw,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz,gx,gy,gz\n";
        epochs<<std::setprecision(17)<<"epoch,timestamp_ns,raw_cam0_ns,raw_cam1_ns,exposure_cam0_ns,exposure_cam1_ns,status,backend_version,tracked_cam0,tracked_cam1,inliers_cam0,inliers_cam1,observations_cam0,observations_cam1,bearing_rms_rad,pose_condition,frontend_dp_m,frontend_dR_rad,visual_reason\n";
        windows<<std::setprecision(17)<<"epoch,version,base_version,accepted,window,landmarks,cam0_obs,cam1_obs,visual_before_rad,visual_after_rad,imu_before_whitened,imu_after_whitened,bias_before_whitened,bias_after_whitened,cost_before,cost_after,max_dp_m,max_dR_rad,max_dv,max_dbg,max_dba,runtime_ms,iterations,marginalizations,prior_rank,prior_condition,reason\n";
        size_t processed=0,tracking=0,prediction=0,accepted=0,rejected=0;int64_t recording_start=0;
        bool dense_online=false;std::vector<DenseRecord> pending_dense;
        auto backend_result=[&](std::shared_ptr<const HiltiBackendResult> r) {
            if(!r)return;
            const auto& d=r->diagnostics;if(d.accepted)++accepted;else ++rejected;
            windows<<processed<<','<<r->version<<','<<r->base_version<<','<<d.accepted<<','<<d.window_size<<','<<d.landmarks
                <<','<<d.camera_observations[0]<<','<<d.camera_observations[1]<<','<<d.visual_before_rad<<','<<d.visual_after_rad
                <<','<<d.imu_before<<','<<d.imu_after<<','<<d.bias_before<<','<<d.bias_after<<','<<d.cost_before<<','<<d.cost_after
                <<','<<d.pose_update_m<<','<<d.rotation_update_rad<<','<<d.velocity_update<<','<<d.gyro_update<<','<<d.accel_update
                <<','<<d.runtime_ms<<','<<d.iterations<<','<<d.marginalizations<<','<<d.prior_rank<<','<<d.prior_condition<<',';
            Quote(windows,d.reason);windows<<'\n';
        };
        HiltiImageEpochReader reader(dataset,calibration);HiltiImageEpoch image;
        const auto started=std::chrono::steady_clock::now();
        while(reader.Next(image)) {
            if(!processed)recording_start=image.cameras[0].timestamp_ns;
            const auto r=frontend.AddHiltiEpoch(image);++processed;backend_result(r.backend_update);
            DenseRecord dense;dense.epoch=processed;dense.frame_index=processed-1;dense.timestamp_ns=image.timestamp_ns;
            dense.status=r.status;dense.backend_version=r.backend_version;dense.frame=r.frame;
            if(!dense_online)pending_dense.push_back(dense);
            if(r.initialized_now||r.status==HiltiFrontendStatus::InitializationFailed) {
                std::ofstream initialization(prefix+"_initialization.txt");
                initialization<<"initialization_epoch="<<processed<<"\ninitialization_frame_index="<<processed-1
                    <<"\ninitialization_timestamp_ns="<<image.timestamp_ns<<"\ninitialization_raw_cam0_ns="<<image.cameras[0].timestamp_ns
                    <<"\ninitialization_elapsed_s="<<std::setprecision(17)<<double(image.cameras[0].timestamp_ns-recording_start)*1e-9<<'\n';
                if(r.startup_publication)
                    initialization<<"initialization_window_start_frame="<<r.startup_publication->start_frame
                        <<"\ninitialization_window_end_frame="<<r.startup_publication->end_frame<<'\n';
                if(r.startup_publication && r.startup_publication->recovery) {
                    const auto& recovery=*r.startup_publication->recovery;
                    initialization<<"startup_recovery_attempted="<<recovery.attempted
                        <<"\nstartup_recovery_accepted="<<recovery.accepted
                        <<"\nstartup_recovery_reason="<<recovery.reason
                        <<"\nstartup_recovery_first_eligible_frame="<<recovery.first_eligible_frame
                        <<"\nstartup_recovery_eligible_buffered_frames="<<recovery.eligible_buffered_frames
                        <<"\nstartup_recovery_eligible_published_frames="<<recovery.eligible_published_frames
                        <<"\nstartup_recovery_eligible_coverage_percent="<<recovery.eligible_coverage_percent
                        <<"\nstartup_recovery_visual_refined_frames="<<recovery.visual_refined_frames
                        <<"\nstartup_recovery_imu_only_frames="<<recovery.imu_only_frames
                        <<"\nstartup_recovery_overlap_frames="<<recovery.overlap_frames
                        <<"\nstartup_recovery_timestamp_mismatches="<<recovery.timestamp_mismatches
                        <<"\nstartup_recovery_duplicate_frames="<<recovery.duplicate_frames
                        <<"\nstartup_recovery_nonfinite_states="<<recovery.nonfinite_states
                        <<"\nstartup_recovery_overlap_max_position_disagreement_m="<<recovery.overlap_max_position_disagreement_m
                        <<"\nstartup_recovery_overlap_max_rotation_disagreement_rad="<<recovery.overlap_max_rotation_disagreement_rad
                        <<"\nstartup_recovery_overlap_max_velocity_disagreement_m_s="<<recovery.overlap_max_velocity_disagreement_m_s
                        <<"\nstartup_recovery_overlap_max_bias_disagreement="<<recovery.overlap_max_bias_disagreement<<'\n';
                }
                initialization<<"initialization_attempts="<<frontend.HiltiStartupAttempts().size()<<'\n';
                for(const auto& attempt:frontend.HiltiStartupAttempts())
                    initialization<<"attempt="<<attempt.attempt<<','<<attempt.start_frame<<','<<attempt.end_frame<<','
                        <<VioInitializationStatusName(attempt.status)<<','<<attempt.recoverable<<','
                        <<attempt.raw_cam0_start_ns<<','<<attempt.raw_cam0_end_ns<<','
                        <<attempt.imu_sample_start_ns<<','<<attempt.imu_sample_end_ns<<','<<attempt.imu_samples<<','
                        <<attempt.reason<<'\n';
                WriteHiltiInitialization(initialization,frontend.HiltiInitialization());
                WriteHiltiInitialization(std::cout,frontend.HiltiInitialization());std::cout.flush();
                if(!initialization)throw std::runtime_error("cannot publish initialization diagnostics");
                if(r.status==HiltiFrontendStatus::InitializationFailed) {
                    for(const auto& record:pending_dense)DenseRow(local,record,record.status,nullptr,Vec3::Constant(NAN));
                    local.flush();return 3;
                }
                if(!r.startup_publication)throw std::logic_error("successful initialization lacks startup publication");
                std::ofstream startup(prefix+"_startup_states.csv");
                startup<<std::setprecision(17)<<"frame,timestamp_ns,px,py,pz,qx,qy,qz,qw,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz,gx,gy,gz\n";
                const auto& init=frontend.HiltiInitialization();
                for(size_t i=0;i<init.states.size();++i) {
                    startup<<r.startup_publication->start_frame+i<<','<<r.startup_publication->reference_timestamps_ns.at(i);
                    State(startup,init.states[i],init.gravity_w);startup<<'\n';
                }
                if(!startup)throw std::runtime_error("cannot publish startup states");
                std::ofstream raw_startup(prefix+"_startup_trajectory.csv");
                raw_startup<<std::setprecision(17)<<"frame,raw_cam0_timestamp_ns,px,py,pz,qx,qy,qz,qw,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz,gx,gy,gz\n";
                for(size_t i=0;i<init.states.size();++i) {
                    raw_startup<<r.startup_publication->start_frame+i<<','<<r.startup_publication->raw_cam0_timestamps_ns.at(i);
                    State(raw_startup,init.states[i],init.gravity_w);raw_startup<<'\n';
                }
                if(!raw_startup)throw std::runtime_error("cannot publish raw-timestamp startup trajectory");
                if(r.startup_publication->recovery) {
                    std::ofstream recovery(prefix+"_startup_recovery.csv");
                    recovery<<std::setprecision(17)<<"frame_index,raw_cam0_timestamp_ns,reference_timestamp_ns,"
                        "initializer_overlap,mode,metric_landmark_observations,pose_only_residual_rad,pose_condition,"
                        "imu_forward_position_error_m,imu_forward_rotation_error_rad,imu_forward_velocity_error_m_s,"
                        "initializer_position_disagreement_m,initializer_rotation_disagreement_rad,"
                        "initializer_velocity_disagreement_m_s,initializer_bias_disagreement,"
                        "px,py,pz,qx,qy,qz,qw,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz,gx,gy,gz\n";
                    for(const auto& frame:r.startup_publication->recovery->frames) {
                        recovery<<frame.frame_index<<','<<frame.raw_cam0_timestamp_ns<<','<<frame.reference_timestamp_ns
                            <<','<<frame.initializer_overlap<<','<<HiltiStartupRecoveryModeName(frame.mode)<<','
                            <<frame.metric_landmark_observations<<','<<frame.pose_only_residual_rad<<','<<frame.pose_condition
                            <<','<<frame.imu_forward_position_error_m<<','<<frame.imu_forward_rotation_error_rad<<','
                            <<frame.imu_forward_velocity_error_m_s<<','<<frame.initializer_position_disagreement_m<<','
                            <<frame.initializer_rotation_disagreement_rad<<','<<frame.initializer_velocity_disagreement_m_s
                            <<','<<frame.initializer_bias_disagreement;
                        State(recovery,frame.state,init.gravity_w);recovery<<'\n';
                    }
                    if(!recovery)throw std::runtime_error("cannot publish startup recovery diagnostics");
                }
            }
            if(r.status==HiltiFrontendStatus::Tracking)++tracking;
            if(r.status==HiltiFrontendStatus::PredictionOnly)++prediction;
            const auto nav=r.frame->HiltiNavigation();
            if(r.global_navigation) {
                const auto& g=*r.global_navigation;NavState output;
                output.position_wb=g.body_pose_global.translation();output.rotation_wb=g.body_pose_global.so3();
                output.velocity_w=g.velocity_global;output.bias=g.bias_body;
                
            }
            if(dense_online)DenseRow(local,dense,r.status,nav.state.get(),frontend.HiltiInitialization().gravity_w);
            if(r.initialized_now) {
                const auto& publication=*r.startup_publication;
                std::map<size_t,const NavState*> recovered;
                if(publication.recovery && publication.recovery->accepted)
                    for(const auto& frame:publication.recovery->frames)
                        if(!frame.initializer_overlap)recovered.emplace(frame.frame_index,&frame.state);
                for(const auto& record:pending_dense) {
                    const auto restored=recovered.find(record.frame_index);
                    if(restored!=recovered.end()) {
                        DenseRow(local,record,HiltiFrontendStatus::Initialized,restored->second,
                            frontend.HiltiInitialization().gravity_w);
                    } else if(publication.start_frame && record.frame_index>=publication.start_frame &&
                       record.frame_index<=publication.end_frame) {
                        const auto& state=publication.states.at(record.frame_index-publication.start_frame);
                        DenseRow(local,record,HiltiFrontendStatus::Initialized,&state,frontend.HiltiInitialization().gravity_w);
                    } else {
                        const auto buffered=record.frame->HiltiNavigation();
                        DenseRow(local,record,record.status,record.status==HiltiFrontendStatus::Buffering?nullptr:buffered.state.get(),
                            frontend.HiltiInitialization().gravity_w);
                    }
                }
                pending_dense.clear();dense_online=true;
            }
            epochs<<processed<<','<<image.timestamp_ns<<','<<image.cameras[0].timestamp_ns<<','<<image.cameras[1].timestamp_ns
                <<','<<image.camera_imu_timestamps_ns[0]<<','<<image.camera_imu_timestamps_ns[1]<<','<<HiltiFrontendStatusName(r.status)
                <<','<<r.backend_version<<','<<r.tracking[0].successfully_tracked<<','<<r.tracking[1].successfully_tracked
                <<','<<r.visual.inliers[0]<<','<<r.visual.inliers[1]<<','<<r.visual.observations[0]<<','<<r.visual.observations[1]
                <<','<<r.visual.bearing_rms_rad<<','<<r.visual.condition<<','<<r.visual.correction_m<<','<<r.visual.correction_rad<<',';
            Quote(epochs,r.reason);epochs<<'\n';
            if(r.backend_job_submitted)backend_result(frontend.PollHiltiBackend(true));
            const auto published=r.frame->HiltiNavigation();
            
            if(processed%100==0) {local.flush();epochs.flush();windows.flush();const auto s=loops->Statistics();
                std::cerr<<"progress epochs="<<processed<<" loops="<<s.accepted_constraints<<" backend_accepted="<<accepted<<" backend_rejected="<<rejected<<'\n';}
        }
        backend_result(frontend.PollHiltiBackend(true));frontend.FlushHiltiLoopClosing();backend->Stop();loops->Stop();
        frontend.FlushHiltiGlobalTrajectory();global->Stop();
        const auto batch=global->Corrections().Snapshot();const auto update=global->LastUpdate();
        if(!batch->has_solution||!update)throw std::runtime_error("no final pose graph output");
        Keyframes(prefix+"_local_keyframes.csv",batch->local_keyframes);Keyframes(prefix+"_global_keyframes.csv",batch->global_keyframes);
        {std::ofstream graph(prefix+"_graph.txt");WritePoseGraphInput(graph,loops->GraphSnapshot());if(!graph)throw std::runtime_error("cannot write graph");}
        double dp=0,dR=0;
        for(size_t k=0;k<batch->local_keyframes.size();++k) {
            const auto& a=batch->local_keyframes[k].body_pose;const auto& b=batch->global_keyframes[k].body_pose;
            dp=std::max(dp,(b.translation()-a.translation()).norm());dR=std::max(dR,(a.so3().inverse()*b.so3()).log().norm());
        }
        const auto s=loops->Statistics();const auto statistics=global->Statistics();
        std::cout<<std::setprecision(17)<<"epochs_processed="<<processed<<"\ntracking_epochs="<<tracking<<"\nprediction_only_epochs="<<prediction
            <<"\naccepted_backend_windows="<<accepted<<"\nrejected_backend_windows="<<rejected
            <<"\naccepted_loop_constraints="<<s.accepted_constraints<<"\nretained_loops="<<update->result.retained_loop_edges
            <<"\ngraph_vertices="<<batch->global_keyframes.size()<<"\nfailed_loop_inputs="<<s.failed_inputs
            <<"\nfailed_global_jobs="<<statistics.failed_jobs<<"\nglobal_correction_versions="<<batch->version
            <<"\npose_graph_max_position_correction_m="<<dp<<"\npose_graph_max_rotation_correction_rad="<<dR
            <<"\nfull_run_runtime_s="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<'\n';
        if(!local||!epochs||!windows)throw std::runtime_error("runtime artifact write failed");
        local.close();epochs.close();windows.close();
        if(s.failed_inputs||statistics.failed_jobs||processed!=dataset.GetStreamInfo(SensorStream::Cam0).count)return 4;
        std::ofstream complete(prefix+"_complete.txt");
        complete<<"epochs_processed="<<processed<<"\nrecorded_cam0_images="<<dataset.GetStreamInfo(SensorStream::Cam0).count
            <<"\nrecording_start_ns="<<recording_start<<"\naccepted_loop_constraints="<<s.accepted_constraints<<'\n';
        if(!complete)throw std::runtime_error("cannot write full-run completion marker");
        return 0;
    }catch(const std::exception& error){std::cerr<<"error: "<<error.what()<<'\n';return 1;}
}
