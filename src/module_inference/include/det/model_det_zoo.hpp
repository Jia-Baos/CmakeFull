#pragma once

#include <map>

#include "../model_det.hpp"

using GetDetModelFunc = std::shared_ptr<DetModel>(*)(const std::string&);

/**
 * @brief 所有检测模型集合
 * 获取模型时，会从这里根据推理框架名称和模型名称寻找匹配的模型
 * 实现新模型后，需要在model_zoo加入对应的获取模型接口
 */
extern std::map<ModelSpec, GetDetModelFunc> det_model_zoo;