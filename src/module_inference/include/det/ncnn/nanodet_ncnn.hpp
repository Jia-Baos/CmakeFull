#pragma once
#include "../../model_det.hpp"

class NanoDetNCNN  : public DetModel{
public:
        static std::shared_ptr<DetModel> GetModel(const std::string &config_path);
};
