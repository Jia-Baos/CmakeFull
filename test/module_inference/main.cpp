#include <iostream>
#include <typeinfo>
#include <opencv2/opencv.hpp>
#include "model_base.hpp"
#include "model_det.hpp"
#include "nanodet_mnn.hpp"

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};


static const std::vector<std::string> class_name = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", 
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", 
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
     "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", 
     "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
      "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
      "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

static const int color_list[80][3] = {
    //{255 ,255 ,255}, //bg
    { 216, 82, 24 },
    { 236, 176, 31 },
    { 125, 46, 141 },
    { 118, 171, 47 },
    { 76, 189, 237 },
    { 238, 19, 46 },
    { 76, 76, 76 },
    { 153, 153, 153 },
    { 255, 0, 0 },
    { 255, 127, 0 },
    { 190, 190, 0 },
    { 0, 255, 0 },
    { 0, 0, 255 },
    { 170, 0, 255 },
    { 84, 84, 0 },
    { 84, 170, 0 },
    { 84, 255, 0 },
    { 170, 84, 0 },
    { 170, 170, 0 },
    { 170, 255, 0 },
    { 255, 84, 0 },
    { 255, 170, 0 },
    { 255, 255, 0 },
    { 0, 84, 127 },
    { 0, 170, 127 },
    { 0, 255, 127 },
    { 84, 0, 127 },
    { 84, 84, 127 },
    { 84, 170, 127 },
    { 84, 255, 127 },
    { 170, 0, 127 },
    { 170, 84, 127 },
    { 170, 170, 127 },
    { 170, 255, 127 },
    { 255, 0, 127 },
    { 255, 84, 127 },
    { 255, 170, 127 },
    { 255, 255, 127 },
    { 0, 84, 255 },
    { 0, 170, 255 },
    { 0, 255, 255 },
    { 84, 0, 255 },
    { 84, 84, 255 },
    { 84, 170, 255 },
    { 84, 255, 255 },
    { 170, 0, 255 },
    { 170, 84, 255 },
    { 170, 170, 255 },
    { 170, 255, 255 },
    { 255, 0, 255 },
    { 255, 84, 255 },
    { 255, 170, 255 },
    { 42, 0, 0 },
    { 84, 0, 0 },
    { 127, 0, 0 },
    { 170, 0, 0 },
    { 212, 0, 0 },
    { 255, 0, 0 },
    { 0, 42, 0 },
    { 0, 84, 0 },
    { 0, 127, 0 },
    { 0, 170, 0 },
    { 0, 212, 0 },
    { 0, 255, 0 },
    { 0, 0, 42 },
    { 0, 0, 84 },
    { 0, 0, 127 },
    { 0, 0, 170 },
    { 0, 0, 212 },
    { 0, 0, 255 },
    { 0, 0, 0 },
    { 36, 36, 36 },
    { 72, 72, 72 },
    { 109, 109, 109 },
    { 145, 145, 145 },
    { 182, 182, 182 },
    { 218, 218, 218 },
    { 0, 113, 188 },
    { 80, 182, 188 },
    { 127, 127, 0 },
};

int resize_uniform(cv::Mat &src, cv::Mat &dst, cv::Size dst_size, object_rect &effect_area)
{
    int w = src.cols;
    int h = src.rows;
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    // std::cout << "src: (" << h << ", " << w << ")" << std::endl;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    float ratio_src = w * 1.0 / h;
    float ratio_dst = dst_w * 1.0 / dst_h;

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst) {
        tmp_w = dst_w;
        tmp_h = floor((dst_w * 1.0 / w) * h);
    } else if (ratio_src < ratio_dst) {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    } else {
        cv::resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    // std::cout << "tmp: (" << tmp_h << ", " << tmp_w << ")" << std::endl;
    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));

    if (tmp_w != dst_w) {
        int index_w = floor((dst_w - tmp_w) / 2.0);
        // std::cout << "index_w: " << index_w << std::endl;
        for (int i = 0; i < dst_h; i++) {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);    // 居中放置
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    } else if (tmp_h != dst_h) {
        int index_h = floor((dst_h - tmp_h) / 2.0);
        // std::cout << "index_h: " << index_h << std::endl;
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);    // 居中放置
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    } else {
        printf("error\n");
    }
    // cv::imwrite("dst", dst);
    // cv::waitKey(0);
    return 0;
}

void draw_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes, object_rect effect_roi, std::string save_path="None")
{
    static const char* class_names[] = { "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                        "train", "truck", "boat", "traffic light", "fire hydrant",
                                        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                        "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                        "baseball glove", "skateboard", "surfboard", "tennis racket",
                                        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                        "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                        "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                        "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();
    int src_w = image.cols;
    int src_h = image.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;


    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(color_list[bbox.label][0], color_list[bbox.label][1], color_list[bbox.label][2]);
        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", bbox.label, bbox.score,
        //    bbox.x1, bbox.y1, bbox.x2, bbox.y2);

        cv::rectangle(image, cv::Rect(cv::Point((bbox.x1 - effect_roi.x) * width_ratio, (bbox.y1 - effect_roi.y) * height_ratio),
                                      cv::Point((bbox.x2 - effect_roi.x) * width_ratio, (bbox.y2 - effect_roi.y) * height_ratio)), color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (bbox.x1 - effect_roi.x) * width_ratio;
        int y = (bbox.y1 - effect_roi.y) * height_ratio - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    if (save_path == "None")
    {
        cv::imshow("image", image);
    }
    else
    {
        cv::imwrite(save_path, image);
        std::cout << save_path << std::endl;
    }
}


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
    object_rect effect_roi;

    const int width = model->GetInputWidth();
    const int height = model->GetInputHeight();
    resize_uniform(src, resized_img, cv::Size(width, height), effect_roi);

    std::shared_ptr<DLOutput> out = model->Infer(resized_img);

    // 需要根据任务类型手动转换到子类，否则无法获取结果
    if (out->m_type == ModelType::kDetection) {
        auto derivate_ptr = std::dynamic_pointer_cast<DetOutput>(out);
        std::cout << ModelTypeToString(derivate_ptr->m_type) << std::endl;
        std::cout << "res size: " << derivate_ptr->m_res.size() << std::endl;
    }
    else if (out->m_type == ModelType::kInstanceSegmentation) {
    }

    std::vector<BoxInfo> results;
    auto derivate_ptr = std::dynamic_pointer_cast<NanoDetMNN>(model);
    results = derivate_ptr->m_result_list;
    draw_bboxes(src, results, effect_roi);

    cv::waitKey();
    return 0;
}
