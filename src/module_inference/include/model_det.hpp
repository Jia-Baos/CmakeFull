#pragma once

#include "./model_base.hpp"

struct ObjectRect {
    int x;      // 原图按比例 resize 后拷贝到模型输入尺寸图像的起始像素点的x坐标（向右）
    int y;      // 原图按比例 resize 后拷贝到模型输入尺寸图像的起始像素点的y坐标（向下）
    int width;  // 原图按比例 resize 后的宽度
    int height; // 原图按比例 resize 后的高度
};

struct DetResult {
    int cls;         // 类别
    float score;     // 置信度
    cv::Rect2f bbox; // 检测框(Point, Point)
};

class DetOutput : public DLOutput {
public:
    DetOutput() : DLOutput(ModelType::kDetection) {}
    virtual ~DetOutput() {}
    std::vector<DetResult> m_res{};
};

class DetModel : public DLModel {
public:
    virtual std::shared_ptr<DLOutput> Infer(const cv::Mat &img);
    virtual std::shared_ptr<DetOutput> Detect(const cv::Mat &img) = 0;
    static std::shared_ptr<DetModel> GetModel(const std::string &config_path);

    ObjectRect m_object_rect{};
    
protected:
    DetModel() : DLModel(ModelType::kDetection) {}
    virtual ~DetModel() {}
};

/**********************************************************************/
/****************************识别任务通用函数*****************************/
/**********************************************************************/

int ResizeUniform(const cv::Mat &src, cv::Mat &dst, const cv::Size &dst_size, ObjectRect &effect_area);

/**
 * 在头文件中使用 static 关键字修饰 class_name，其作用是：
 * - 每个包含该头文件的源文件都会有一份独立的 class_name 副本。
 * - 避免链接时出现“多重定义”错误（multiple definition），因为 static 使其具有内部链接属性，仅在当前编译单元（.cpp 文件）可见。
 * - 适用于常量查表、颜色表等无需跨编译单元共享的全局数据。

 * 简而言之，static 让 class_name 只在本编译单元内可见，防止头文件中全局变量重复定义导致链接错误。
 */

static const std::vector<std::string> class_name = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

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