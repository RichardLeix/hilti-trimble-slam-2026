#ifndef MYSLAM_DATASET_H
#define MYSLAM_DATASET_H
#include "myslam/camera.h"
#include "myslam/common_include.h"
#include "myslam/frame.h"

namespace myslam {

/// Common interface for dataset initialization, cameras, and stereo frames.
class Dataset {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Dataset> Ptr;
    virtual ~Dataset() = default;

    /// 初始化，返回是否成功
    virtual bool Init() = 0;

    /// create and return the next frame containing the stereo images
    virtual Frame::Ptr NextFrame() = 0;

    /// get camera by id
    virtual Camera::Ptr GetCamera(int camera_id) const = 0;
};
}  // namespace myslam

#endif
