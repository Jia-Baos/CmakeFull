#include <iostream>
#include <typeinfo>
#include <opencv2/opencv.hpp>
#include "model_base.hpp"
#include "model_det.hpp"
#include "nanodet_openvino.hpp"

static cv::Mat DrawBoxes(const cv::Mat &img, const std::vector<NanoDet::BoxInfo> &bboxes, ObjectRect effect_roi)
{
    cv::Mat res_img = img.clone();

    int src_w = res_img.cols;
    int src_h = res_img.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;

    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;

    for (size_t i = 0; i < bboxes.size(); i++) {
        const NanoDet::BoxInfo &bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(kColorList[bbox.label][0], kColorList[bbox.label][1], kColorList[bbox.label][2]);

        cv::rectangle(res_img, cv::Rect(cv::Point((bbox.x1 - effect_roi.x) * width_ratio, (bbox.y1 - effect_roi.y) * height_ratio), cv::Point((bbox.x2 - effect_roi.x) * width_ratio, (bbox.y2 - effect_roi.y) * height_ratio)), color);

        // char text[256];
        // sprintf(text, "%s %.1f%%", class_name[bbox.label], bbox.score * 100);

        std::string text = kClassName[bbox.label] + " " + std::to_string(bbox.score);
        std::cout << "class name: " << kClassName[bbox.label] << ", score: " << bbox.score << std::endl;

        int base_line = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &base_line);

        int x = (bbox.x1 - effect_roi.x) * width_ratio;
        int y = (bbox.y1 - effect_roi.y) * height_ratio - label_size.height - base_line;
        if (y < 0) {
            y = 0;
        }

        if (x + label_size.width > res_img.cols) {
            x = res_img.cols - label_size.width;
        }

        cv::rectangle(res_img, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + base_line)), color, -1);
        cv::putText(res_img, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    return res_img;
}

int main()
{
    std::cout << "=============================================================" << std::endl;
    std::cout << "test DLModel inference api" << std::endl;
    std::cout << "=============================================================" << std::endl;

    // 获取模型
    std::string img_path = "/home/jia-baos/Project-Cpp/CmakeFull/imgs/000252.jpg";
    std::string config_path = "/home/jia-baos/Project-Cpp/CmakeFull/config/nanodet_openvino.yaml";
    std::shared_ptr<DLModel> model = DLModel::GetModel(config_path);
    if (!model) {
        return 0;
    }

    // 运行推理
    cv::Mat src = cv::imread(img_path);

    std::cout << "src size: " << src.size() << std::endl;

    cv::Mat resized_img;
    ObjectRect effect_roi;
    std::shared_ptr<DLOutput> out = model->Infer(src);

    // 需要根据任务类型手动转换到子类，否则无法获取结果
    if (out->m_type == ModelType::kDetection) {
        auto derivate_ptr = std::dynamic_pointer_cast<DetOutput>(out);
        std::cout << "ModelType: " << ModelTypeToString(derivate_ptr->m_type) << std::endl;
        // std::cout << "res size: " << derivate_ptr->m_res.size() << std::endl;
    } else if (out->m_type == ModelType::kInstanceSegmentation) {
    }

    auto derivate_ptr = std::dynamic_pointer_cast<NanoDetOPENVINO>(model);
    auto results = derivate_ptr->m_result_list;

    cv::Mat res = DrawBoxes(src, results, derivate_ptr->m_object_rect);
    cv::imshow("res", res);
    cv::waitKey();
    return 0;
}
