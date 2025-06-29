#pragma once

#include "../../model_det.hpp"

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <string>
#include <vector>
#include <memory>

class NanoDetOPENVINO : public DetModel {
public:
    NanoDetOPENVINO(const std::string &config_path);
    virtual ~NanoDetOPENVINO();
    virtual std::shared_ptr<DetOutput> Detect(const cv::Mat &img);
    static std::shared_ptr<DetModel> GetModel(const std::string &config_path);

    void decode_infer(const float* pred, const std::vector<NanoDet::CenterPrior> &center_priors, const float threshold, std::vector<std::vector<NanoDet::BoxInfo>> &results);
    NanoDet::BoxInfo disPred2Bbox(const float *dfl_det, const int label, const float score, const int x, const int y, const int stride);
    void nms(std::vector<NanoDet::BoxInfo> &input_boxes, const float nms_thresh);

    // modify these parameters to the same with your config if you want to use your own model
    // int input_size[2] = { 416, 416 };          // input height and width
    int num_class = 80;                           // number of classes. 80 for COCO
    int reg_max = 7;                              // `reg_max` set in the training config. Default: 7.
    std::vector<int> strides = { 8, 16, 32, 64 }; // strides of the multi-level feature.

    std::string input_name = "data";
    std::string output_name = "output";

    std::vector<NanoDet::BoxInfo> m_result_list;

private:
    ov::CompiledModel network_;
    ov::InferRequest infer_request_;

    int m_thread_num;
    int m_class_num;

    float m_nms_threshold;
    float m_confidence_threshold;

    const float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
    const float norm_vals[3] = { 0.017429f, 0.017507f, 0.017125f };
};
