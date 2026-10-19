#include "myslam/camera.h"
#include <cmath>

namespace myslam {

Camera::Camera() {
}

Vec3 Camera::world2camera(const Vec3 &p_w, const SE3 &T_c_w) {
    return pose_ * T_c_w * p_w;
}

Vec3 Camera::camera2world(const Vec3 &p_c, const SE3 &T_c_w) {
    return T_c_w.inverse() * pose_inv_ * p_c;
}

Vec2 Camera::camera2pixel(const Vec3 &p_c) {
    return Vec2(
            fx_ * p_c(0, 0) / p_c(2, 0) + cx_,
            fy_ * p_c(1, 0) / p_c(2, 0) + cy_
    );
}

Vec3 Camera::pixel2camera(const Vec2 &p_p, double depth) {
    return Vec3(
            (p_p(0, 0) - cx_) * depth / fx_,
            (p_p(1, 0) - cy_) * depth / fy_,
            depth
    );
}

Vec2 Camera::world2pixel(const Vec3 &p_w, const SE3 &T_c_w) {
    return camera2pixel(world2camera(p_w, T_c_w));
}

Vec3 Camera::pixel2world(const Vec2 &p_p, const SE3 &T_c_w, double depth) {
    return camera2world(pixel2camera(p_p, depth), T_c_w);
}

bool Camera::IsValidPixel(const Vec2 &pixel) const {
    return pixel.allFinite() && std::isfinite(fx_) && std::isfinite(fy_) &&
           std::isfinite(cx_) && std::isfinite(cy_) && fx_ > 0 && fy_ > 0;
}

bool Camera::Project(const Vec3 &point, Vec2 &pixel) const {
    if (!point.allFinite() || point.z() <= 0) return false;
    const Vec2 result(fx_ * (point.x() / point.z()) + cx_,
                      fy_ * (point.y() / point.z()) + cy_);
    if (!IsValidPixel(result)) return false;
    pixel = result;
    return true;
}

bool Camera::Unproject(const Vec2 &pixel, Vec3 &bearing) const {
    if (!IsValidPixel(pixel)) return false;
    const Vec3 ray((pixel.x() - cx_) / fx_, (pixel.y() - cy_) / fy_, 1);
    if (!ray.allFinite()) return false;
    const Vec3 scaled = ray / ray.cwiseAbs().maxCoeff();
    const Vec3 result = scaled / scaled.norm();
    if (!result.allFinite()) return false;
    bearing = result;
    return true;
}

}
