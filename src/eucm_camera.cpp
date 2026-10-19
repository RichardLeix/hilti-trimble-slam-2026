#include "myslam/eucm_camera.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace myslam {
namespace {
const double kDomainTolerance = 16 * std::numeric_limits<double>::epsilon();
}

EucmCamera::EucmCamera(double alpha, double beta, double fx, double fy,
                       double cx, double cy, int width, int height,
                       const SE3 &T_cam_reference, const cv::Mat &invalid_mask)
    : Camera(fx, fy, cx, cy, 0, T_cam_reference),
      alpha_(alpha), beta_(beta), width_(width), height_(height) {
    if (!std::isfinite(alpha) || alpha < 0 || alpha > 1 ||
        !std::isfinite(beta) || beta <= 0 ||
        !Camera::IsValidPixel(Vec2(cx, cy)) || width <= 0 || height <= 0) {
        throw std::invalid_argument("invalid EUCM intrinsics or resolution");
    }
    if (!invalid_mask.empty()) {
        if (invalid_mask.type() != CV_8UC1 || invalid_mask.cols != width ||
            invalid_mask.rows != height) {
            throw std::invalid_argument("EUCM mask must be grayscale and match the resolution");
        }
        invalid_mask_ = invalid_mask.clone();
    }
}

// EUCM equations and domains: Usenko et al. 2018, Sec. 2.3, Eqs. 8-9, 16-23.
// https://cvg.cit.tum.de/_media/spezial/bib/usenko18double-sphere.pdf
bool EucmCamera::IsValidPixel(const Vec2 &pixel) const {
    if (!Camera::IsValidPixel(pixel) || pixel.x() < 0 || pixel.y() < 0 ||
        pixel.x() >= width_ || pixel.y() >= height_) return false;
    const double mx = (pixel.x() - cx_) / fx_;
    const double my = (pixel.y() - cy_) / fy_;
    const double r2 = mx * mx + my * my;
    const double discriminant = 1 - (2 * alpha_ - 1) * beta_ * r2;
    // Exclude the exact inverse-domain boundary, where projection is not unique.
    if (!std::isfinite(r2) || !std::isfinite(discriminant) ||
        discriminant <= kDomainTolerance) return false;
    if (invalid_mask_.empty()) return true;
    // Integer coordinates denote pixel centers; sample the nearest mask pixel.
    const int u = std::min(width_ - 1, int(std::floor(pixel.x() + 0.5)));
    const int v = std::min(height_ - 1, int(std::floor(pixel.y() + 0.5)));
    return invalid_mask_.at<uint8_t>(v, u) == 0;
}

bool EucmCamera::Project(const Vec3 &point, Vec2 &pixel) const {
    if (!point.allFinite()) return false;
    const double scale = point.cwiseAbs().maxCoeff();
    if (scale == 0) return false;
    const Vec3 p = point / scale;  // Projection is homogeneous; avoid overflow.
    const double d = std::hypot(std::sqrt(beta_) * std::hypot(p.x(), p.y()), p.z());
    const double w = alpha_ <= 0.5 ? alpha_ / (1 - alpha_) : (1 - alpha_) / alpha_;
    if (p.z() + w * d <= kDomainTolerance * d) return false;
    const double denominator = alpha_ * d + (1 - alpha_) * p.z();
    if (!std::isfinite(denominator) || denominator <= 0) return false;
    const Vec2 result(fx_ * (p.x() / denominator) + cx_,
                      fy_ * (p.y() / denominator) + cy_);
    if (!IsValidPixel(result)) return false;
    pixel = result;
    return true;
}

bool EucmCamera::Unproject(const Vec2 &pixel, Vec3 &bearing) const {
    if (!IsValidPixel(pixel)) return false;
    const double mx = (pixel.x() - cx_) / fx_;
    const double my = (pixel.y() - cy_) / fy_;
    const double r2 = mx * mx + my * my;
    const double root = std::sqrt(1 - (2 * alpha_ - 1) * beta_ * r2);
    const double mz = (1 - beta_ * alpha_ * alpha_ * r2) /
                      (alpha_ * root + 1 - alpha_);
    const Vec3 ray(mx, my, mz);
    if (!ray.allFinite() || ray.cwiseAbs().maxCoeff() == 0) return false;
    const Vec3 scaled = ray / ray.cwiseAbs().maxCoeff();
    const Vec3 result = scaled / scaled.norm();
    if (!result.allFinite()) return false;
    bearing = result;
    return true;
}

Vec2 EucmCamera::camera2pixel(const Vec3 &point) {
    Vec2 pixel;
    if (!Project(point, pixel)) throw std::domain_error("point outside EUCM valid domain");
    return pixel;
}

Vec3 EucmCamera::pixel2camera(const Vec2 &pixel, double depth) {
    Vec3 bearing;
    if (!std::isfinite(depth) || depth <= 0 || !Unproject(pixel, bearing)) {
        throw std::domain_error("pixel or range outside EUCM valid domain");
    }
    return bearing * depth;
}

}  // namespace myslam
