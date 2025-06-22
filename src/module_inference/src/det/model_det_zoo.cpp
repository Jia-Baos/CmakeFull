#include "../../include/det/model_det_zoo.hpp"

std::map<ModelSpec, GetDetModelFunc> det_model_zoo = {
#ifdef USE_MNN
    { ModelSpec{"nanodet", "mnn"}, &NanoDetMNN::GetModel },
#endif
#ifdef USE_NCNN
    { ModelSpec{"nanodet", "ncnn"}, &NanoDetNCNN::GetModel }
#endif
};