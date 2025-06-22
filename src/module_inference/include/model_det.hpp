#pragma once

#include "./model_base.hpp"

struct DetResult {
    int cls;       // 类别
    float score;   // 置信度
    cv::Rect bbox; // 检测框
};

class DetOutput : public DLOutput {
public:
    DetOutput() : DLOutput(ModelType::kDetection) {}
    std::vector<DetResult> m_res{};
};

class DetModel : public DLModel {
public:
    virtual std::shared_ptr<DLOutput> Infer(const cv::Mat &img);
    virtual std::shared_ptr<DetOutput> Detect(const cv::Mat &img) = 0;
    static std::shared_ptr<DetModel> GetModel(const std::string &config_path);

protected:
    DetModel() : DLModel(ModelType::kDetection) {}
    virtual ~DetModel() {}
};
