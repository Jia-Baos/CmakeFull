# 使用说明

## DLModel层
```c++
// 获取模型接口
static std::shared_ptr<DLModel> getModel(std::string config_path);

// 推理接口
std::shared_ptr<DLOutput> infer(cv::Mat& img);
```
DLOutput为输出结果的抽象类，使用者需要根据实际的类型对DLOutput结果做解析。

## 任务层
以目标检测模型为例

```c++
// 获取模型接口
static std::shared_ptr<DetModel> getModel(std::string config_path);

// 推理接口
std::shared_ptr<DetOutput> detect(cv::Mat& img);
```
其中DetOutput为目标检测输出类型
```c++
// 目标检测输出
struct DetOutput : public DLOutput
{
    DetOutput():
        DLOutput(Detection){};

    std::vector<DetResult> res;
};
// 目标检测结果
struct DetResult
{
    int cls;        // 类别
    float conf;     // 置信度
    cv::Rect bbox;  // 检测框
};
```
