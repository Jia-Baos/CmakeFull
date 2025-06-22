#include <iostream>
#include "model_det.hpp"
#include "../include/det/model_det_zoo.hpp"

std::shared_ptr<DLOutput> DetModel::Infer(const cv::Mat &img)
{
    return this->Detect(img);
}

std::shared_ptr<DetModel> DetModel::GetModel(const std::string &config_path)
{
    cv::FileStorage config(config_path, cv::FileStorage::READ);
    std::string task_name;
    config["task"] >> task_name;
    std::string model_name;
    config["model"] >> model_name;
    std::string framework_name;
    config["framework"] >> framework_name;
    config.release();

    if (task_name != "det") {
        std::cout << "Not a detection model, please check" << std::endl;
        return std::shared_ptr<DetModel>(nullptr);
    }

    ModelSpec spec = ModelSpec{ model_name, framework_name };
    if (det_model_zoo.count(spec) == 0) {
        std::cout << "Unsupported model, please check model name and inference engine" << std::endl;
        return std::shared_ptr<DetModel>(nullptr);
    }
    else {
        std::shared_ptr<DetModel> model = det_model_zoo[spec](config_path);
        return model;
    }
}
