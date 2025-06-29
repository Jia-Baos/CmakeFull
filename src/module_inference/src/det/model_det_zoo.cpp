#include "../../include/det/model_det_zoo.hpp"

#ifdef USE_MNN
#include "../../include/det/mnn/nanodet_mnn.hpp"
#endif

#ifdef USE_NCNN
#include "../../include/det/ncnn/nanodet_ncnn.hpp"
#endif

#ifdef USE_OPENVINO
#include "../../include/det/openvino/nanodet_openvino.hpp"
#endif

std::map<ModelSpec, GetDetModelFunc> det_model_zoo = {
#ifdef USE_MNN
    { ModelSpec{"nanodet", "mnn"}, &NanoDetMNN::GetModel },
#endif
#ifdef USE_NCNN
    { ModelSpec{"nanodet", "ncnn"}, &NanoDetNCNN::GetModel },
#endif
#ifdef USE_OPENVINO
    { ModelSpec{"nanodet", "openvino"}, &NanoDetOPENVINO::GetModel }
#endif
};