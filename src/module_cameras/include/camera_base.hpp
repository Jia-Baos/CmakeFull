#pragma once
#include <atomic>
#include <cstdint>
#include <string>
#include <optional>
#include <mutex>
#include <shared_mutex>
#include <opencv2/opencv.hpp>

#ifdef USE_BOOST
#include <boost/thread.hpp>
#endif

namespace cgu
{
#ifdef USE_BOOST
using SCOPE_LOCK = boost::lock_guard<boost::mutex>;
using WRITE_LOCK = boost::unique_lock<boost::shared_mutex>;
using READ_LOCK = boost::shared_lock<boost::shared_mutex>;
#else
using SCOPE_LOCK = std::lock_guard<::std::mutex>;
using WRITE_LOCK = std::unique_lock<::std::shared_mutex>;
using READ_LOCK = std::shared_lock<::std::shared_mutex>;
#endif
} // namespace cgu

struct DataFrame {
    cv::Mat img;
    uint64_t timestamp;
};

class CameraBase {
public:
    CameraBase()
        : m_stop_flag(false) {}
    virtual ~CameraBase() {}

    virtual void SetIP(const std::string &str) = 0;
    virtual void SetSN(const std::string &str) = 0;
    virtual bool Wait4Device() = 0;
    virtual bool InitDevice() = 0;
    virtual void Run() = 0;
    virtual void StartCapture() = 0;
    
    bool Stop()
    {
        m_stop_flag = true;
        while (m_stop_flag) {
            std::cout << "waiting for stop camera" << std::endl;
        }
        std::cout << "stopped camera" << std::endl;
        return true;
    }

    virtual std::optional<DataFrame> GetDataFrame() = 0;

public:
    std::string m_ip{};                     // 网口相机 IP
    std::string m_sn{};                     // USB相机序列号
    DataFrame m_data_frame;                 // 帧数据
    std::shared_mutex m_frame_mutex;        // 帧数据互斥锁
    std::atomic<bool> m_stop_flag{ false }; // 相机禁用 flag
};
