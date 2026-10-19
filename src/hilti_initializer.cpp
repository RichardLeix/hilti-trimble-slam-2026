#include "myslam/hilti_initializer.h"
#include "uncertainty_metric_initializer.h"
#include <iomanip>
#include <memory>

namespace myslam {
namespace {
VioInitializationStatus MetricFailure(const std::string& reason) {
    if (reason.find("excitation") != std::string::npos) return VioInitializationStatus::InsufficientExcitation;
    if (reason.find("rank/conditioning") != std::string::npos) return VioInitializationStatus::IllConditioned;
    if (reason.find("gravity magnitude") != std::string::npos || reason.find("scale") != std::string::npos ||
        reason.find("implausible") != std::string::npos) return VioInitializationStatus::ImplausibleSolution;
    return VioInitializationStatus::AlignmentInconsistent;
}
VioInitializationResult InitializeHiltiStartupImpl(const BootstrapResult* supplied_visual,
    const std::vector<HiltiTrackingResult>* tracks,
    const std::vector<BearingMotionResult>* motions, const Camera& camera,
    const std::vector<ImuSample>& samples, const ImuCalibration& calibration,
    const VioInitializerOptions& options) {
    VioInitializationResult result;
    result.path = "robust_visual_combined_uncertainty_fixed_accel_bias";
    auto fail = [&](VioInitializationStatus status, const std::string& reason) {
        result.status = status; result.message = reason;
        result.states.clear(); result.scale = NAN; result.gravity_w.setConstant(NAN);
        result.initial_velocity_w.setConstant(NAN);
        return result;
    };
    std::string phase = "robust visual bootstrap";
    try {
        VioInitializer native(options); // Existing option validation and gyro fit.
        std::unique_ptr<BootstrapResult> owned_visual;
        if (!supplied_visual) {
            if (!tracks || !motions) throw std::invalid_argument("missing startup visual inputs");
            owned_visual.reset(new BootstrapResult(
                BuildMultiFrameStartup(*tracks, *motions, camera, calibration, options)));
            supplied_visual = owned_visual.get();
        }
        const auto& visual = *supplied_visual;
        const auto& window = visual.window;
        result.frames_used = window.frames.size();
        if (!window.positions_in_common_scale) return fail(VioInitializationStatus::VisualStructureUnavailable,
            phase + ": " + window.translation_failure);
        phase = "gyro bias estimation";
        VioImuIntervals zero_bias;
        for (size_t i = 1; i < window.frames.size(); ++i) zero_bias.push_back(PreintegrateImu(samples,
            window.frames[i-1].timestamp_ns, window.frames[i].timestamp_ns, calibration));
        const auto gyro = native.EstimateGyroBias(window, zero_bias, samples, calibration, camera.pose());
        result.duration_s = gyro.duration_s; result.gyro_condition = gyro.gyro_condition;
        result.rotation_rms_rad = gyro.rotation_rms_rad; result.max_rotation_error_rad = gyro.max_rotation_error_rad;
        result.bias = gyro.bias; result.gyro_bias_estimated = gyro.gyro_bias_estimated;
        if (!gyro.gyro_bias_estimated) return fail(gyro.status, phase + ": " + gyro.message);

        // A stages 1/2 supply ONLY the declared numerical seed for feasible GLS.
        // No accel-bias release and no publication/fallback to this candidate.
        phase = "fixed-bias covariance seed";
        const auto seed = staged_metric::Run(window, samples, calibration, camera.pose(), result.bias.gyroscope,
            staged_metric::Weighting::Heuristic, options, staged_metric::FinalStage::FixedAccelBias);
        result.acceleration_excitation = seed.excitation;
        if (!seed.success) return fail(MetricFailure(seed.reason), phase + ": " + seed.reason);

        phase = "visual BA covariance";
        const auto visual_covariance = visual_uncertainty::Compute(visual, options.max_bearing_rms_rad);
        auto& d = result.uncertainty;
        d.combined = true; d.visual_covariance_valid = visual_covariance.usable;
        d.gauge_dimensions = visual_covariance.gauge_dimensions; d.anchor_frame = window.anchor_frame;
        d.visual_rank = visual_covariance.rank; d.visual_columns = visual_covariance.conditional_columns;
        d.deficient_landmarks = visual_covariance.deficient_landmarks; d.residual_dof = visual_covariance.dof;
        d.bearing_variance = visual_covariance.bearing_variance; d.visual_condition = visual_covariance.condition;
        d.gauge_null_leakage = visual_covariance.scale_null_leakage;
        d.anchor_constraint_error = visual_covariance.anchor_constraint_error;
        if (!visual_covariance.usable) return fail(VioInitializationStatus::UncertaintyInvalid,
            phase + ": " + visual_covariance.reason);

        phase = "combined visual+IMU covariance alignment";
        const auto combined = uncertainty_metric::Run(visual, visual_covariance, samples, calibration,
            camera.pose(), result.bias.gyroscope, seed.stages[1].scale, options);
        d.reweighting_iterations = combined.reweighting_iterations; d.converged = combined.converged;
        d.relative_scale_step = combined.relative_scale_step;
        const auto& c = combined.covariance;
        d.covariance_condition = c.raw_condition; d.normalized_covariance_condition = c.normalized_condition;
        d.cross_interval_covariance_norm = c.cross_interval_norm;
        d.position_visual_variance = c.position_visual_variance; d.position_imu_variance = c.position_imu_variance;
        d.velocity_visual_variance = c.velocity_visual_variance; d.velocity_imu_variance = c.velocity_imu_variance;
        if (!combined.metric.success || !combined.converged) return fail(MetricFailure(combined.metric.reason),
            phase + ": " + combined.metric.reason);
        d.whitening_error = (c.whitening*c.combined*c.whitening.transpose()-
            MatXX::Identity(c.combined.rows(), c.combined.cols())).cwiseAbs().maxCoeff();
        const auto& stage = combined.metric.stages[1];
        result.alignment_condition = stage.data_fit.condition; result.alignment_rank = stage.data_fit.rank;
        result.alignment_columns = stage.data_a.cols();
        result.position_rms_m = stage.position_rms; result.velocity_rms_m_s = stage.velocity_rms;
        d.scale_std_unit_noise = std::sqrt(stage.data_fit.covariance_unit_noise(stage.scale_column, stage.scale_column));
        d.scale_std_residual_scaled = d.scale_std_unit_noise*std::sqrt(stage.data_fit.residual_variance);
        d.velocity_scale_correlation = staged_metric::Correlation(stage, false, false);
        result.gravity_w = stage.gravity; result.scale = stage.scale;
        result.states = stage.states; result.initial_velocity_w = result.states.front().velocity_w;
        result.bias.accelerometer = stage.accel_bias; result.accel_bias_estimated = false;
        result.status = VioInitializationStatus::Success;
        result.message = "robust visual BA + combined visual/IMU uncertainty alignment; accelerometer bias fixed at zero";
        return result;
    } catch (const std::exception& error) {
        const bool covariance = phase.find("covariance") != std::string::npos && phase != "fixed-bias covariance seed";
        return fail(covariance ? VioInitializationStatus::UncertaintyInvalid : VioInitializationStatus::InvalidInput,
            phase + ": " + error.what());
    }
}
}
VioInitializationResult InitializeHiltiStartup(const std::vector<HiltiTrackingResult>& tracks,
    const std::vector<BearingMotionResult>& motions, const Camera& camera,
    const std::vector<ImuSample>& samples, const ImuCalibration& calibration,
    const VioInitializerOptions& options) {
    return InitializeHiltiStartupImpl(nullptr, &tracks, &motions, camera, samples, calibration, options);
}

void WriteHiltiInitialization(std::ostream& out, const VioInitializationResult& r) {
    const auto precision = out.precision(); out << std::setprecision(17);
    auto vector = [&](const char* name, const Vec3& value) {out<<name<<'='<<value.x()<<','<<value.y()<<','<<value.z()<<'\n';};
    out<<"initializer_path="<<r.path<<"\ninitialization_success="<<r.Success()
       <<"\ninitialization_status="<<VioInitializationStatusName(r.status)<<"\ninitialization_reason="<<r.message
       <<"\nscale_metres_per_visual_unit="<<r.scale<<"\naccel_bias_estimated="<<r.accel_bias_estimated
       <<"\ngyro_bias_estimated="<<r.gyro_bias_estimated
       <<"\ngravity_magnitude_m_s2="<<r.gravity_w.norm()
       <<"\naccel_bias_state="<<(r.accel_bias_estimated ? "estimated_during_startup" : "fixed_zero_prior")<<'\n';
    vector("gravity_w_m_s2",r.gravity_w);vector("initial_velocity_w_m_s",r.initial_velocity_w);
    if (!r.states.empty()) vector("current_initialized_velocity_w_m_s",r.states.back().velocity_w);
    vector("gyro_bias_rad_s",r.bias.gyroscope);vector("accel_bias_m_s2",r.bias.accelerometer);
    out<<"gyro_condition="<<r.gyro_condition<<"\nalignment_condition="<<r.alignment_condition
       <<"\nalignment_rank="<<r.alignment_rank<<"\nalignment_columns="<<r.alignment_columns
       <<"\nposition_rms_m="<<r.position_rms_m<<"\nvelocity_rms_m_s="<<r.velocity_rms_m_s<<'\n';
    const auto& d=r.uncertainty;
#define DIAGNOSTIC(field) out<<"initializer_" #field "="<<d.field<<'\n'
    DIAGNOSTIC(combined);DIAGNOSTIC(visual_covariance_valid);DIAGNOSTIC(converged);
    DIAGNOSTIC(gauge_dimensions);DIAGNOSTIC(anchor_frame);DIAGNOSTIC(visual_rank);DIAGNOSTIC(visual_columns);
    DIAGNOSTIC(deficient_landmarks);DIAGNOSTIC(residual_dof);DIAGNOSTIC(bearing_variance);DIAGNOSTIC(visual_condition);
    DIAGNOSTIC(gauge_null_leakage);DIAGNOSTIC(anchor_constraint_error);DIAGNOSTIC(covariance_condition);
    DIAGNOSTIC(normalized_covariance_condition);DIAGNOSTIC(cross_interval_covariance_norm);DIAGNOSTIC(whitening_error);
    DIAGNOSTIC(reweighting_iterations);DIAGNOSTIC(relative_scale_step);DIAGNOSTIC(scale_std_unit_noise);
    DIAGNOSTIC(scale_std_residual_scaled);DIAGNOSTIC(velocity_scale_correlation);
    DIAGNOSTIC(position_visual_variance);DIAGNOSTIC(position_imu_variance);
    DIAGNOSTIC(velocity_visual_variance);DIAGNOSTIC(velocity_imu_variance);
#undef DIAGNOSTIC
    out.precision(precision);
}
} // namespace myslam
