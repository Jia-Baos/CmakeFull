#pragma once

#include <libobsensor/ObSensor.hpp>
#include <libobsensor/hpp/Error.hpp>
#include <libobsensor/hpp/StreamProfile.hpp>
#include <libobsensor/hpp/Context.hpp>
#include <libobsensor/hpp/Device.hpp>
#include <libobsensor/hpp/Frame.hpp>
#include <libobsensor/hpp/Pipeline.hpp>

#include "camera_base.hpp"

class OrbbecCamera : public CameraBase {
public:
    OrbbecCamera();
    virtual ~OrbbecCamera();
    void SetIP(const std::string &str) override;
    void SetSN(const std::string &str) override;
    bool Wait4Device() override;
    bool InitDevice() override;
    void Run() override;

    std::optional<cv::Mat> GetImg() override;

private:
    std::shared_ptr<ob::Device> m_device{ nullptr };
    std::shared_ptr<ob::Pipeline> m_pipe{ nullptr };
    std::shared_ptr<ob::Config> m_config{ nullptr };
    ob::PointCloudFilter m_point_cloud;
    cv::Mat m_intrinsic{ cv::Mat(3, 3, CV_32F) };
    cv::Mat m_distort{ cv::Mat(1, 8, CV_32F) };
};
