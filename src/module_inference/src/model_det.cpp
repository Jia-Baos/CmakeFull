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
    } else {
        std::shared_ptr<DetModel> model = det_model_zoo[spec](config_path);
        return model;
    }
}

/**********************************************************************/
/****************************识别任务通用函数*****************************/
/**********************************************************************/

int ResizeUniform(const cv::Mat &src, cv::Mat &dst, const cv::Size &dst_size, ObjectRect &effect_area)
{
    const int src_w = src.cols;
    const int src_h = src.rows;
    const int dst_w = dst_size.width;
    const int dst_h = dst_size.height;

    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    float ratio_src = src_w * 1.0f / src_h;
    float ratio_dst = dst_w * 1.0f / dst_h;

    int keep_ratio_w = 0;
    int keep_ratio_h = 0;
    if (ratio_src > ratio_dst) {
        keep_ratio_w = dst_w;
        keep_ratio_h = std::floor((dst_w * 1.0 / src_w) * src_h);
    } else if (ratio_src < ratio_dst) {
        keep_ratio_h = dst_h;
        keep_ratio_w = std::floor((dst_h * 1.0 / src_h) * src_w);
    } else {
        cv::resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    cv::Mat src_keep_ratio;
    cv::resize(src, src_keep_ratio, cv::Size(keep_ratio_w, keep_ratio_h));

    if (keep_ratio_h != dst_h) {
        int index_h = std::floor((dst_h - keep_ratio_h) / 2.0);

        memcpy(dst.data + index_h * dst_w * 3, src_keep_ratio.data, keep_ratio_w * keep_ratio_h * 3); // 居中放置
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = keep_ratio_w;
        effect_area.height = keep_ratio_h;
    } else if (keep_ratio_w != dst_w) {
        int index_w = std::floor((dst_w - keep_ratio_w) / 2.0);

        for (int i = 0; i < dst_h; i++) {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, src_keep_ratio.data + i * keep_ratio_w * 3, keep_ratio_w * 3); // 居中放置
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = keep_ratio_w;
        effect_area.height = keep_ratio_h;
    }

    return 0;
}
