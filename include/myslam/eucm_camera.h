#ifndef MYSLAM_EUCM_CAMERA_H
#define MYSLAM_EUCM_CAMERA_H

#include "myslam/camera.h"

namespace myslam {

// Extended Unified Camera Model. Intrinsic order: alpha, beta, fx, fy, cx, cy.
// Uses the original Hilti calibration rather than its equidistant approximation.
class EucmCamera : public Camera {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EucmCamera(double alpha, double beta, double fx, double fy, double cx,
               double cy, int width, int height, const SE3 &T_cam_reference,
               const cv::Mat &invalid_mask = cv::Mat());

    bool Project(const Vec3 &point, Vec2 &pixel) const override;
    bool Unproject(const Vec2 &pixel, Vec3 &bearing) const override;
    bool IsValidPixel(const Vec2 &pixel) const override;

    Vec2 camera2pixel(const Vec3 &point) override;
    // For EUCM, depth is radial range along the unit bearing, not pinhole Z.
    Vec3 pixel2camera(const Vec2 &pixel, double depth = 1) override;

    double Alpha() const { return alpha_; }
    double Beta() const { return beta_; }
    int Width() const { return width_; }
    int Height() const { return height_; }

   private:
    double alpha_, beta_;
    int width_, height_;
    cv::Mat invalid_mask_;  // Official OpenVINS mask: zero valid, nonzero invalid.
};

}  // namespace myslam

#endif  // MYSLAM_EUCM_CAMERA_H
