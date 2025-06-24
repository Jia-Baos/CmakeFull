#include <iostream>
#include <typeinfo>
#include <opencv2/opencv.hpp>
#include "model_base.hpp"
#include "model_det.hpp"
#include "nanodet_mnn.hpp"

int main()
{
    std::cout << "=============================================================" << std::endl;
    std::cout << "test DLModel inference api" << std::endl;
    std::cout << "=============================================================" << std::endl;
    
    // 获取模型
    std::string img_path = "/home/jia-baos/Project-Cpp/CmakeFull/imgs/000252.jpg";
    std::string config_path = "/home/jia-baos/Project-Cpp/CmakeFull/config/nanodet_mnn.yaml";
    std::shared_ptr<DLModel> model = DLModel::GetModel(config_path);
    if (!model)
    {
        return 0;
    }

    // 运行推理
    cv::Mat src = cv::imread(img_path);

    std::cout << "src size: " << src.size() << std::endl;

    cv::Mat resized_img;
    ObjectRect effect_roi;

    // const int width = model->GetInputWidth();
    // const int height = model->GetInputHeight();
    // ResizeUniform(src, resized_img, cv::Size(width, height), effect_roi);

    std::shared_ptr<DLOutput> out = model->Infer(src);

    // 需要根据任务类型手动转换到子类，否则无法获取结果
    if (out->m_type == ModelType::kDetection) {
        auto derivate_ptr = std::dynamic_pointer_cast<DetOutput>(out);
        std::cout << "ModelType: " << ModelTypeToString(derivate_ptr->m_type) << std::endl;
        // std::cout << "res size: " << derivate_ptr->m_res.size() << std::endl;
    }
    else if (out->m_type == ModelType::kInstanceSegmentation) {
    }

    auto derivate_ptr = std::dynamic_pointer_cast<NanoDetMNN>(model);
    auto results = derivate_ptr->m_result_list;

    cv::Mat res = DrawBoxes(src, results, derivate_ptr->m_object_rect);

    cv::imshow("res", res);
    cv::waitKey();
    return 0;
}
