#pragma once

#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

/**
 * @brief 模型类型枚举
 */
enum class ModelType {
    kDetection,
    kInstanceSegmentation
};

inline std::string ModelTypeToString(ModelType type) {
    switch(type) {
        case ModelType::kDetection: return "kDetection";
        case ModelType::kInstanceSegmentation: return "kInstanceSegmentation";
        default: return "Unknown";
    }
}
/**
 *   通过函数指针返回具体任务实例
 */
struct ModelSpec {
    std::string model_name;
    std::string framework_name;

    bool operator<(const ModelSpec &other) const
    {
        return std::tie(framework_name, model_name) < std::tie(other.framework_name, other.model_name);
    }

    bool operator==(const ModelSpec &other) const
    {
        return model_name == other.model_name && framework_name == other.framework_name;
    }
};

class DLOutput {
public:
    DLOutput(ModelType type) : m_type(type) {}
    virtual ~DLOutput() {}
    ModelType m_type;
};

class DLModel {
public:
    virtual std::shared_ptr<DLOutput> Infer(const cv::Mat &img) = 0;
    static std::shared_ptr<DLModel> GetModel(const std::string &config_path);

    bool IsValid();
    int GetInputWidth();
    int GetInputHeight();
    ModelType GetModelType();

protected:
    DLModel(ModelType model_type) : m_type(model_type), m_valid(true) {}
    virtual ~DLModel() {};

    ModelType m_type{};   // 模型类型
    bool m_valid{};       // 模型是否有效
    int m_input_width{};  // 模型输入宽度
    int m_input_height{}; // 模型输入高度
};
