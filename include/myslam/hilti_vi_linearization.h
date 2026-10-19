#ifndef MYSLAM_HILTI_VI_LINEARIZATION_H
#define MYSLAM_HILTI_VI_LINEARIZATION_H
#include "myslam/hilti_backend.h"

namespace myslam {
struct HiltiNormalEquations {
    MarginalizationVariables variables; // Chronological states first, camera-local landmarks second.
    MatXX hessian;
    VecX gradient;

};
MarginalizationVariables HiltiGraphVariables(const HiltiBackendJob& job);
MarginalizationVariables HiltiPriorVariables(const HiltiBackendJob& job);
Eigen::Matrix<double,15,15> HiltiImuWhitening(const ImuPreintegration& delta);
// Empty connected set means all live factors. A nonempty set selects factors
// incident to outgoing variables; the ENTIRE old prior is included once because
// it is replaced, never duplicated. No optimizer damping or hard fix is added.
HiltiNormalEquations LinearizeHiltiGraph(const HiltiBackendJob& job,const HiltiBackendOptions& options,
    const std::set<MarginalizationKey>& connected = {});
} // namespace myslam
#endif
