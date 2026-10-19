#include "myslam/hilti_dataset.h"
#include "myslam/eucm_camera.h"

#include <dirent.h>
#include <sqlite3.h>
#include <sys/stat.h>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace myslam {
namespace {

const std::array<const char*, 3> kTopics{{
    "/cam0/image_raw/compressed", "/cam1/image_raw/compressed", "/imu/data_raw"}};
const std::array<const char*, 3> kTypes{{
    "sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/CompressedImage",
    "sensor_msgs/msg/Imu"}};

size_t StreamIndex(SensorStream stream) {
    const size_t index = static_cast<size_t>(stream);
    if (index >= kTopics.size()) throw std::out_of_range("invalid sensor stream");
    return index;
}

struct DatabaseCloser {
    void operator()(sqlite3* db) const { sqlite3_close(db); }
};
struct StatementCloser {
    void operator()(sqlite3_stmt* statement) const { sqlite3_finalize(statement); }
};
using Database = std::unique_ptr<sqlite3, DatabaseCloser>;
using Statement = std::unique_ptr<sqlite3_stmt, StatementCloser>;

void CheckSql(int result, sqlite3* db) {
    if (result != SQLITE_OK) throw std::runtime_error(sqlite3_errmsg(db));
}

Statement Prepare(sqlite3* db, const char* sql) {
    sqlite3_stmt* raw = nullptr;
    const int result = sqlite3_prepare_v2(db, sql, -1, &raw, nullptr);
    Statement statement(raw);
    CheckSql(result, db);
    return statement;
}

int Step(sqlite3_stmt* statement) {
    const int result = sqlite3_step(statement);
    if (result != SQLITE_ROW && result != SQLITE_DONE) {
        throw std::runtime_error(sqlite3_errmsg(sqlite3_db_handle(statement)));
    }
    return result;
}

std::string Text(sqlite3_stmt* statement, int column) {
    const auto* value = sqlite3_column_text(statement, column);
    if (!value) throw std::runtime_error("null rosbag topic metadata");
    return reinterpret_cast<const char*>(value);
}

std::string BagFile(const std::string& path) {
    struct stat status;
    if (stat(path.c_str(), &status) != 0) {
        throw std::runtime_error("cannot access rosbag path: " + path);
    }
    if (S_ISREG(status.st_mode)) return path;
    if (!S_ISDIR(status.st_mode)) throw std::runtime_error("invalid rosbag path");

    std::unique_ptr<DIR, int (*)(DIR*)> directory(opendir(path.c_str()), closedir);
    if (!directory) throw std::runtime_error("cannot open rosbag directory: " + path);
    std::vector<std::string> files;
    errno = 0;
    while (auto* entry = readdir(directory.get())) {
        const std::string name = entry->d_name;
        if (name.size() >= 4 && name.substr(name.size() - 4) == ".db3") {
            const std::string file = path + "/" + name;
            struct stat file_status;
            if (stat(file.c_str(), &file_status) != 0) {
                throw std::runtime_error("cannot access rosbag file: " + file);
            }
            if (S_ISREG(file_status.st_mode)) files.push_back(file);
        }
        errno = 0;
    }
    if (errno != 0) throw std::runtime_error("cannot list rosbag directory");
    if (files.size() != 1) {
        throw std::runtime_error("expected exactly one .db3 file in the bag directory");
    }
    return files.front();
}

double CalibrationNumber(const cv::FileNode& node) {
    if ((!node.isReal() && !node.isInt()) || !std::isfinite(double(node))) {
        throw std::runtime_error("calibration contains a missing or nonfinite numeric value");
    }
    return double(node);
}

Camera::Ptr LoadCalibratedCamera(const cv::FileNode& node, const std::string& directory) {
    if (!node.isMap() || std::string(node["camera_model"]) != "eucm" ||
        std::string(node["distortion_model"]) != "none") {
        throw std::runtime_error("Hilti calibration must specify camera_model: eucm, distortion_model: none");
    }
    const auto intrinsics = node["intrinsics"];
    const auto distortion = node["distortion_coeffs"];
    const auto resolution = node["resolution"];
    if (!intrinsics.isSeq() || intrinsics.size() != 6 || !distortion.isSeq() ||
        distortion.size() != 0 || !resolution.isSeq() || resolution.size() != 2 ||
        !resolution[0].isInt() || !resolution[1].isInt()) {
        throw std::runtime_error("invalid EUCM intrinsics, distortion coefficients or resolution");
    }
    std::array<double, 6> parameters;
    for (size_t i = 0; i < parameters.size(); ++i) {
        parameters[i] = CalibrationNumber(intrinsics[int(i)]);
    }
    const auto extrinsic = node["T_cam_imu"];
    if (!extrinsic.isSeq() || extrinsic.size() != 4) {
        throw std::runtime_error("T_cam_imu must be a 4x4 rigid transform");
    }
    Eigen::Matrix4d transform;
    for (int row = 0; row < 4; ++row) {
        if (!extrinsic[row].isSeq() || extrinsic[row].size() != 4) {
            throw std::runtime_error("T_cam_imu must be a 4x4 rigid transform");
        }
        for (int col = 0; col < 4; ++col) {
            transform(row, col) = CalibrationNumber(extrinsic[row][col]);
        }
    }
    const Mat33 rotation = transform.block<3, 3>(0, 0);
    const Mat33 orthogonality_error = rotation.transpose() * rotation - Mat33::Identity();
    const double determinant = rotation.determinant();
    if ((transform.row(3) - Eigen::RowVector4d(0, 0, 0, 1)).norm() > 1e-12 ||
        !orthogonality_error.allFinite() || !std::isfinite(determinant) ||
        orthogonality_error.norm() >= 1e-10 || std::abs(determinant - 1) >= 1e-10) {
        throw std::runtime_error("T_cam_imu is not a proper rigid transform");
    }
    if (!node["mask"].isString() || std::string(node["mask"]).empty()) {
        throw std::runtime_error("Hilti calibration requires an official camera mask");
    }
    const std::string mask_name = node["mask"];
    const std::string mask_file = mask_name.front() == '/' ? mask_name : directory + "/" + mask_name;
    const cv::Mat mask = cv::imread(mask_file, cv::IMREAD_GRAYSCALE);
    if (mask.empty()) throw std::runtime_error("cannot read camera mask: " + mask_file);
    const SE3 T_cam_imu(SO3(rotation), transform.block<3, 1>(0, 3));
    return Camera::Ptr(new EucmCamera(parameters[0], parameters[1], parameters[2],
        parameters[3], parameters[4], parameters[5], int(resolution[0]),
        int(resolution[1]), T_cam_imu, mask));
}

// CDR1 alignment is relative to the body AFTER its four-byte encapsulation.
// Mirrors the validated Python CdrReader; supports either byte order.
class CdrReader {
   public:
    CdrReader(const void* data, size_t size) {
        if (!data || size < 4) throw std::runtime_error("truncated CDR encapsulation");
        const auto* bytes = static_cast<const uint8_t*>(data);
        const unsigned representation = (unsigned(bytes[0]) << 8) | bytes[1];
        if (representation > 1) throw std::runtime_error("unsupported CDR representation");
        little_endian_ = representation == 1;
        data_ = bytes + 4;
        size_ = size - 4;
    }

    const uint8_t* Take(size_t count) {
        if (count > size_ - offset_) throw std::runtime_error("truncated CDR field");
        const auto* result = data_ + offset_;
        offset_ += count;
        return result;
    }

    uint32_t Uint32() {
        Align(4);
        return static_cast<uint32_t>(Bits(4));
    }

    double Double() {
        Align(8);
        const uint64_t bits = Bits(8);
        double value;
        static_assert(sizeof(value) == sizeof(bits), "CDR requires 64-bit doubles");
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    void SkipDoubles(size_t count) {
        Align(8);
        Take(count * 8);
    }

    std::string String() {
        const uint32_t length = Uint32();
        const auto* bytes = Take(length);
        if (length == 0 || bytes[length - 1] != 0) {
            throw std::runtime_error("CDR string must include its terminating NUL");
        }
        return std::string(reinterpret_cast<const char*>(bytes), length - 1);
    }

    int64_t Timestamp() {
        const uint32_t bits = Uint32();
        int32_t seconds;
        std::memcpy(&seconds, &bits, sizeof(seconds));
        const uint32_t nanoseconds = Uint32();
        if (nanoseconds >= 1000000000U) {
            throw std::runtime_error("invalid header nanoseconds");
        }
        return int64_t(seconds) * 1000000000LL + nanoseconds;
    }

    void Finish() const {
        if (offset_ != size_) throw std::runtime_error("unexpected trailing CDR data");
    }

   private:
    void Align(size_t alignment) { Take((alignment - offset_ % alignment) % alignment); }

    uint64_t Bits(size_t count) {
        const auto* bytes = Take(count);
        uint64_t result = 0;
        for (size_t i = 0; i < count; ++i) {
            const size_t shift = little_endian_ ? i : count - i - 1;
            result |= uint64_t(bytes[i]) << (shift * 8);
        }
        return result;
    }

    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    size_t offset_ = 0;
    bool little_endian_ = true;
};

SensorPacket Decode(SensorStream stream, int64_t id, int64_t bag_timestamp,
                    const void* data, size_t size, bool keep_image) {
    CdrReader reader(data, size);
    SensorPacket packet;
    packet.stream = stream;
    packet.message_id = id;
    packet.bag_timestamp_ns = bag_timestamp;
    packet.timestamp_ns = reader.Timestamp();
    std::string frame_id = reader.String();
    if (stream == SensorStream::Imu) {
        packet.imu.timestamp_ns = packet.timestamp_ns;
        packet.imu.bag_timestamp_ns = bag_timestamp;
        packet.imu.frame_id = std::move(frame_id);
        reader.SkipDoubles(13);  // Orientation quaternion and covariance.
        for (auto& value : packet.imu.angular_velocity) value = reader.Double();
        reader.SkipDoubles(9);  // Angular velocity covariance.
        for (auto& value : packet.imu.linear_acceleration) value = reader.Double();
        reader.SkipDoubles(9);  // Linear acceleration covariance.
    } else {
        packet.image.frame_id = std::move(frame_id);
        packet.image.format = reader.String();
        const uint32_t length = reader.Uint32();
        const auto* bytes = reader.Take(length);
        if (keep_image) packet.image.data.assign(bytes, bytes + length);
    }
    reader.Finish();
    return packet;
}

void Observe(HiltiStreamInfo& info, const SensorPacket& packet) {
    if (info.count == 0) {
        info.first_timestamp_ns = packet.timestamp_ns;
        info.first_bag_timestamp_ns = packet.bag_timestamp_ns;
    } else {
        info.duplicate_timestamps += packet.timestamp_ns == info.last_timestamp_ns;
        info.backward_steps += packet.timestamp_ns < info.last_timestamp_ns;
        info.bag_duplicate_timestamps += packet.bag_timestamp_ns == info.last_bag_timestamp_ns;
        info.bag_backward_steps += packet.bag_timestamp_ns < info.last_bag_timestamp_ns;
    }
    ++info.count;
    info.last_timestamp_ns = packet.timestamp_ns;
    info.last_bag_timestamp_ns = packet.bag_timestamp_ns;
    info.header_bag_mismatches += packet.timestamp_ns != packet.bag_timestamp_ns;
}

}  // namespace

struct HiltiDataset::Impl {
    struct Entry {
        int64_t id;
        SensorStream stream;
        int64_t timestamp_ns;
        int64_t bag_timestamp_ns;
    };

    Database db;
    Statement lookup;
    std::array<int64_t, 3> topic_ids{{0, 0, 0}};
    std::array<HiltiStreamInfo, 3> streams;
    std::vector<Entry> entries;      // Small timestamp/id index, not image payloads.
    std::vector<ImuSample> imu;
    size_t next = 0;
};

HiltiDataset::HiltiDataset(const std::string& rosbag_path)
    : rosbag_path_(rosbag_path) {}

HiltiDataset::~HiltiDataset() = default;

bool HiltiDataset::LoadCameraCalibration(const std::string& calibration_file) {
    try {
        cv::FileStorage file(calibration_file, cv::FileStorage::READ);
        if (!file.isOpened()) throw std::runtime_error("cannot open calibration: " + calibration_file);
        const auto slash = calibration_file.find_last_of('/');
        const std::string directory = slash == std::string::npos ? "." : calibration_file.substr(0, slash);
        std::array<Camera::Ptr, 2> cameras{{
            LoadCalibratedCamera(file["cam0"], directory),
            LoadCalibratedCamera(file["cam1"], directory)}};
        cameras_ = std::move(cameras);
        last_error_.clear();
        return true;
    } catch (const std::exception& error) {
        last_error_ = error.what();
        LOG(ERROR) << "Hilti camera calibration: " << last_error_;
        return false;
    }
}

bool HiltiDataset::Init() {
    impl_.reset();
    last_error_.clear();
    try {
        const std::string file = BagFile(rosbag_path_);
        std::unique_ptr<Impl> state(new Impl);
        sqlite3* raw = nullptr;
        const int opened = sqlite3_open_v2(file.c_str(), &raw, SQLITE_OPEN_READONLY, nullptr);
        state->db.reset(raw);
        if (opened != SQLITE_OK) {
            throw std::runtime_error(raw ? sqlite3_errmsg(raw) : "cannot open rosbag database");
        }
        // Keep one consistent read snapshot for statistics and subsequent packets.
        CheckSql(sqlite3_exec(raw, "BEGIN", nullptr, nullptr, nullptr), raw);
        auto topics = Prepare(raw, "SELECT id, name, type, serialization_format FROM topics");
        std::array<bool, 3> found{{false, false, false}};
        while (Step(topics.get()) == SQLITE_ROW) {
            const std::string name = Text(topics.get(), 1);
            for (size_t i = 0; i < kTopics.size(); ++i) {
                if (name != kTopics[i]) continue;
                if (found[i]) throw std::runtime_error("duplicate topic: " + name);
                if (Text(topics.get(), 2) != kTypes[i] || Text(topics.get(), 3) != "cdr") {
                    throw std::runtime_error("unexpected type or serialization for " + name);
                }
                found[i] = true;
                state->topic_ids[i] = sqlite3_column_int64(topics.get(), 0);
                state->streams[i].topic = name;
                state->streams[i].message_type = kTypes[i];
            }
        }
        for (size_t i = 0; i < found.size(); ++i) {
            if (!found[i]) throw std::runtime_error(std::string("missing topic: ") + kTopics[i]);
        }

        auto messages = Prepare(raw,
            "SELECT id, topic_id, timestamp, data FROM messages "
            "WHERE topic_id IN (?, ?, ?) ORDER BY id");
        for (size_t i = 0; i < state->topic_ids.size(); ++i) {
            CheckSql(sqlite3_bind_int64(messages.get(), int(i + 1), state->topic_ids[i]), raw);
        }
        while (Step(messages.get()) == SQLITE_ROW) {
            const int64_t id = sqlite3_column_int64(messages.get(), 0);
            const int64_t topic_id = sqlite3_column_int64(messages.get(), 1);
            const int64_t bag_timestamp = sqlite3_column_int64(messages.get(), 2);
            const size_t stream_index = size_t(std::find(state->topic_ids.begin(),
                state->topic_ids.end(), topic_id) - state->topic_ids.begin());
            const auto stream = static_cast<SensorStream>(stream_index);
            SensorPacket packet;
            try {
                packet = Decode(stream, id, bag_timestamp,
                    sqlite3_column_blob(messages.get(), 3),
                    size_t(sqlite3_column_bytes(messages.get(), 3)), false);
            } catch (const std::exception& error) {
                throw std::runtime_error("message " + std::to_string(id) + " (" +
                    kTopics.at(stream_index) + "): " + error.what());
            }
            Observe(state->streams.at(stream_index), packet);
            state->entries.push_back({id, stream, packet.timestamp_ns, bag_timestamp});
            if (stream == SensorStream::Imu) state->imu.push_back(std::move(packet.imu));
        }
        std::sort(state->entries.begin(), state->entries.end(),
            [](const Impl::Entry& left, const Impl::Entry& right) {
                if (left.timestamp_ns != right.timestamp_ns) {
                    return left.timestamp_ns < right.timestamp_ns;
                }
                return left.id < right.id;
            });
        std::stable_sort(state->imu.begin(), state->imu.end(),
            [](const ImuSample& left, const ImuSample& right) {
                return left.timestamp_ns < right.timestamp_ns;
            });
        state->lookup = Prepare(raw, "SELECT data FROM messages WHERE id = ?");
        impl_ = std::move(state);
        return true;
    } catch (const std::exception& error) {
        last_error_ = error.what();
        LOG(ERROR) << "HiltiDataset: " << last_error_;
        return false;
    }
}

const HiltiStreamInfo& HiltiDataset::GetStreamInfo(SensorStream stream) const {
    if (!impl_) throw std::logic_error("HiltiDataset is not initialized");
    return impl_->streams.at(StreamIndex(stream));
}

bool HiltiDataset::NextPacket(SensorPacket& packet) {
    if (!impl_) throw std::logic_error("HiltiDataset is not initialized");
    if (impl_->next == impl_->entries.size()) return false;
    const auto& entry = impl_->entries[impl_->next];
    auto* statement = impl_->lookup.get();
    CheckSql(sqlite3_reset(statement), impl_->db.get());
    CheckSql(sqlite3_bind_int64(statement, 1, entry.id), impl_->db.get());
    if (Step(statement) != SQLITE_ROW) throw std::runtime_error("missing indexed rosbag message");
    packet = Decode(entry.stream, entry.id, entry.bag_timestamp_ns,
        sqlite3_column_blob(statement, 0), size_t(sqlite3_column_bytes(statement, 0)), true);
    ++impl_->next;
    return true;
}

void HiltiDataset::Rewind() {
    if (!impl_) throw std::logic_error("HiltiDataset is not initialized");
    impl_->next = 0;
}

std::vector<ImuSample> HiltiDataset::ImuBetween(int64_t begin_ns, int64_t end_ns) const {
    if (!impl_) throw std::logic_error("HiltiDataset is not initialized");
    if (begin_ns >= end_ns) throw std::invalid_argument("IMU interval must have begin < end");
    const auto& samples = impl_->imu;
    const auto begin = std::upper_bound(samples.begin(), samples.end(), begin_ns,
        [](int64_t timestamp, const ImuSample& sample) { return timestamp < sample.timestamp_ns; });
    const auto end = std::lower_bound(begin, samples.end(), end_ns,
        [](const ImuSample& sample, int64_t timestamp) { return sample.timestamp_ns < timestamp; });
    return std::vector<ImuSample>(begin, end);
}

}  // namespace myslam
