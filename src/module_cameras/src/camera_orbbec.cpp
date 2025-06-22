#include "camera_orbbec.hpp"
#include <thread>
#include <chrono>

OrbbecCamera::OrbbecCamera()
{
    ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_FATAL);
}

OrbbecCamera::~OrbbecCamera()
{
    if (m_pipe) {
        m_pipe->stop();
    }
    m_pipe = nullptr;
    m_config = nullptr;
    m_device = nullptr;
}

void OrbbecCamera::SetIP(const std::string &str)
{
    this->m_ip = str;
}

void OrbbecCamera::SetSN(const std::string &str)
{
    this->m_sn = str;
}

bool OrbbecCamera::Wait4Device()
{
    ob::Context ctx;
    auto dev_list = ctx.queryDeviceList();
    int cam_num = dev_list->deviceCount();
    std::stringstream ss;
    ss << "discover camera number is " << cam_num;
    std::cout << ss.str() << std::endl;
    for (int i = 0; i < cam_num; i++) {
        if (!m_device) {
            m_device = dev_list->getDevice(i);
        }

        auto dev_info = m_device->getDeviceInfo();
        auto serial_number = dev_info->serialNumber();
        break;
    }

    if (!m_device) {
        return false;
    }
    return true;
}

bool OrbbecCamera::InitDevice()
{
    m_pipe = std::make_shared<ob::Pipeline>(m_device);
    m_config = std::make_shared<ob::Config>();
    try {
        // 配置rgb摄像头的参数
        auto color_profiles = m_pipe->getStreamProfileList(OB_SENSOR_COLOR);
        std::shared_ptr<ob::VideoStreamProfile> color_profile = nullptr;
        try {
            // 640，480，RGB，30帧
            color_profile = color_profiles->getVideoStreamProfile(640, 480, OB_FORMAT_RGB, 30);
        }
        catch (ob::Error &e) {
            color_profile = color_profiles->getVideoStreamProfile(640, 480, OB_FORMAT_UNKNOWN, 30);
        }
        int rw = color_profile->width();
        int rh = color_profile->height();
        std::stringstream ss;
        ss << "rgb size: " << rh << " * " << rw;
        std::cout << ss.str() << std::endl;
        m_config->enableStream(color_profile);
    }
    catch (ob::Error &e) {
        std::stringstream ss;
        ss << "Current device is not support color sensor!";
        std::cout << ss.str() << std::endl;
    }
    try {
        auto depth_profiles = m_pipe->getStreamProfileList(OB_SENSOR_DEPTH);
        std::shared_ptr<ob::VideoStreamProfile> depth_profile = nullptr;
        try {
            depth_profile = depth_profiles->getVideoStreamProfile(640, 480, OB_FORMAT_Y11, 30);
        }
        catch (ob::Error &e) {
            depth_profile = depth_profiles->getVideoStreamProfile(640, 480, OB_FORMAT_UNKNOWN, 30);
        }

        int pw = depth_profile->width();
        int ph = depth_profile->height();
        std::stringstream ss;
        ss << "depth size : " << ph << " * " << pw;
        std::cout << ss.str() << std::endl;
        m_config->enableStream(depth_profile);
    }
    catch (ob::Error &e) {
        std::stringstream ss;
        ss << "Current device is not support depth sensor!";
        std::cout << ss.str() << std::endl;
    }
    if (m_device->isPropertySupported(OB_PROP_DEPTH_ALIGN_HARDWARE_BOOL, OB_PERMISSION_READ)) {
        m_config->setAlignMode(ALIGN_D2C_HW_MODE);
    } else {
        m_config->setAlignMode(ALIGN_D2C_SW_MODE);
    }

    if (!m_pipe)
        return 0;
    m_pipe->start(m_config);

    auto camera_param = m_pipe->getCameraParam();
    m_point_cloud.setCameraParam(camera_param);
    m_intrinsic.at<float>(0, 0) = camera_param.rgbIntrinsic.fx;
    m_intrinsic.at<float>(0, 1) = 0;
    m_intrinsic.at<float>(0, 2) = camera_param.rgbIntrinsic.cx;
    m_intrinsic.at<float>(1, 0) = 0;
    m_intrinsic.at<float>(1, 1) = camera_param.rgbIntrinsic.fy;
    m_intrinsic.at<float>(1, 2) = camera_param.rgbIntrinsic.cy;
    m_intrinsic.at<float>(2, 0) = 0;
    m_intrinsic.at<float>(2, 1) = 0;
    m_intrinsic.at<float>(2, 2) = 1;
    m_distort.at<float>(0, 0) = camera_param.rgbDistortion.k1;
    m_distort.at<float>(0, 1) = camera_param.rgbDistortion.k2;
    m_distort.at<float>(0, 2) = camera_param.rgbDistortion.p1;
    m_distort.at<float>(0, 3) = camera_param.rgbDistortion.p2;
    m_distort.at<float>(0, 4) = camera_param.rgbDistortion.k3;
    m_distort.at<float>(0, 5) = camera_param.rgbDistortion.k4;
    m_distort.at<float>(0, 6) = camera_param.rgbDistortion.k5;
    m_distort.at<float>(0, 7) = camera_param.rgbDistortion.k6;
    return false;
}

void OrbbecCamera::Run()
{
    while (!m_stop_flag) {

        while (!Wait4Device()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        while (!InitDevice()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        while (!m_stop_flag) {
            auto frame_set = m_pipe->waitForFrames(1000);

            if (frame_set == nullptr) {
                std::stringstream ss;
                ss << "get frame fail!";
                std::cout << ss.str() << std::endl;
                continue;
            }
            if (frame_set != nullptr && frame_set->depthFrame() != nullptr && frame_set->colorFrame() != nullptr) {
                auto depth_value_scale = frame_set->depthFrame()->getValueScale();
                break;
            };
            if (!frame_set || !frame_set->depthFrame() || !frame_set->colorFrame()) {
                std::stringstream ss;
                ss << "frameSet or depthFrame or colorFrame is null";
                std::cout << ss.str() << std::endl;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        auto start = std::chrono::high_resolution_clock::now();
        while (!m_stop_flag) {
            auto frame_set = m_pipe->waitForFrames(100);
            if (frame_set == nullptr) {
                continue;
            }
            if (frame_set != nullptr && frame_set->depthFrame() != nullptr && frame_set->colorFrame() != nullptr) {
                auto depth_value_scale = frame_set->depthFrame()->getValueScale();
                m_point_cloud.setPositionDataScaled(depth_value_scale);
                try {
                    int rw = frame_set->colorFrame()->width();
                    int rh = frame_set->colorFrame()->height();
                    {
                        cgu::WRITE_LOCK(this->m_frame_mutex);
                        m_frame_data.img_rgb = cv::Mat(rh, rw, CV_8UC3, frame_set->colorFrame()->data());
                        m_frame_data.time_stamp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
                    }

                    // calc frame freq
                    auto now = std::chrono::high_resolution_clock::now();
                    auto freq = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
                    start = now;
                    std::stringstream ss;
                    ss << "frame freq is " << freq << " ms";
                    std::cout << ss.str() << std::endl;
                }
                catch (std::exception &e) {
                    std::stringstream ss;
                    ss << "handle frame err " << e.what();
                    std::cout << ss.str() << std::endl;
                }
            } else {
                std::stringstream ss;
                ss << "get color frame or depth frame failed!";
                std::cout << ss.str() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
    }

    m_stop_flag = false;
}

std::optional<cv::Mat> OrbbecCamera::GetImg()
{
    cgu::READ_LOCK(this->m_frame_mutex);
    if (m_frame_data.img_rgb.empty()) {
        return std::nullopt;
    }
    return m_frame_data.img_rgb;
}
