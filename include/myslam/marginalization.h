#ifndef MYSLAM_MARGINALIZATION_H
#define MYSLAM_MARGINALIZATION_H
#include "myslam/nav_state.h"
#include <set>

namespace myslam {
using NavLocalVector = Eigen::Matrix<double,15,1>;
using NavLocalMatrix = Eigen::Matrix<double,15,15>;
// Ordering: pose [right SO(3) rotation, world position], world velocity,
// IMU-axis gyro bias, IMU-axis accel bias. Pose uses SO(3) x R^3, not global
// Euler/quaternion coordinates. R = R0 Exp(dtheta), p = p0 + dp_world.
NavState RetractNavState(const NavState& state, const NavLocalVector& increment);
NavLocalVector NavStateLocalDelta(const NavState& linearization, const NavState& state);
NavLocalMatrix NavStateLocalJacobian(const NavState& linearization, const NavState& state);

struct MarginalizationKey {
    enum class Kind { State, Landmark };
    Kind kind = Kind::State;
    int64_t timestamp_ns = 0;
    int camera_id = 0;
    uint64_t feature_id = 0;
    static MarginalizationKey State(int64_t timestamp);
    static MarginalizationKey Landmark(int camera,uint64_t feature);
    bool operator<(const MarginalizationKey& other) const;
    bool operator==(const MarginalizationKey& other) const;
};
struct MarginalizationVariable {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    MarginalizationKey key;
    NavState state;
    Vec3 point_w = Vec3::Zero();
    int Dimension() const {return key.kind==MarginalizationKey::Kind::State?15:3;}
};
using MarginalizationVariables=std::vector<MarginalizationVariable,Eigen::aligned_allocator<MarginalizationVariable>>;
struct MarginalizationEvaluation {
    VecX delta, residual;
    MatXX jacobian; // Derivative wrt CURRENT local increments, not stored global coordinates.
    MatXX hessian;
    VecX gradient; // J^T r; normal step solves H dx = -gradient.
};
struct MarginalizationPrior {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    MarginalizationVariables linearization;
    MatXX sqrt_information;
    VecX residual_at_linearization;
    MatXX hessian;
    VecX gradient;
    size_t marginalizations = 0, marginalized_landmarks = 0;
    size_t rank = 0;
    double condition = 0; // Nonzero eigenvalue condition after diagonal equilibration.
    bool has_fixed_pose = false;
    int64_t fixed_pose_timestamp_ns = 0; // Only while the original gauge state remains active.
    int Dimension() const {return int(hessian.rows());}
    MarginalizationEvaluation Evaluate(const MarginalizationVariables& current) const;
};
struct MarginalizationInfo {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    MarginalizationVariables variables;
    MatXX hessian; // UNDAMPED Gauss-Newton/robust IRLS normal matrix.
    VecX gradient;
    std::set<MarginalizationKey> eliminate;
    // Exact conditioning of fixed coordinates of a variable being eliminated.
    // Used once to carry the original hard pose gauge; not a large fake weight.
    std::set<int> fixed_coordinates;
    MarginalizationPrior Marginalize(double relative_rank_tolerance=1e-10) const;
};
} // namespace myslam
#endif
