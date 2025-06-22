#include "./model_base.hpp"
#include "./model_det.hpp"
#include "./model_seg.hpp"

std::shared_ptr<DLModel> DLModel::GetModel(const std::string &config_path)
{
    cv::FileStorage config(config_path, cv::FileStorage::READ);
    std::string task_name;
    config["task"] >> task_name;
    config.release();

    if (task_name == "det") {
        return DetModel::GetModel(config_path);
    }
    else {
        std::cout << "Unsupported model type: " << task_name << std::endl;
        return std::shared_ptr<DLModel>(nullptr);
    }
}

bool DLModel::IsValid()
{
    return m_valid;
}

int DLModel::GetInputWidth()
{
    return m_input_width;
}

int DLModel::GetInputHeight()
{
    return m_input_height;
}
ModelType DLModel::GetModelType()
{
    return m_type;
}
