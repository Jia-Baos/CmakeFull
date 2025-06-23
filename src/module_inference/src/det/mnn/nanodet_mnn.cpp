#include "../../../include/det/mnn/nanodet_mnn.hpp"
#include <iostream>
#include <chrono>

float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

template <typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{ 0 };

    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}

void generate_grid_center_priors(const int input_height, const int input_width, std::vector<int> &strides, std::vector<CenterPrior> &center_priors)
{
    for (int i = 0; i < (int)strides.size(); i++) {
        int stride = strides[i];
        int feat_w = ceil((float)input_width / stride);
        int feat_h = ceil((float)input_height / stride);
        for (int y = 0; y < feat_h; y++) {
            for (int x = 0; x < feat_w; x++) {
                CenterPrior ct;
                ct.x = x;
                ct.y = y;
                ct.stride = stride;
                center_priors.push_back(ct);
            }
        }
    }
}

std::string replaceExtension(std::string input, std::string new_ext)
{
    return input.substr(0, input.find_last_of('.')) + new_ext;
}

NanoDetMNN::NanoDetMNN(const std::string &config_path)
{
    // 加载模型，配置模型和推理参数
    cv::FileStorage config_file(config_path, cv::FileStorage::READ);

    config_file["input_height"] >> m_input_height;
    config_file["input_width"] >> m_input_width;
    config_file["num_thread"] >> m_thread_num;
    config_file["nms_threshold"] >> m_nms_threshold;
    config_file["confidence_threshold"] >> m_confidence_threshold;

    m_class_num = config_file["class"].size();
    config_file.release();

    // 权重文件与配置文件同级
    std::string mnn_path = replaceExtension(config_path, ".mnn");
    NanoDet_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    if (!NanoDet_interpreter) {
        std::cout << "Loading model failed!" << std::endl;
        m_valid = false;
        return;
    }

    MNN::ScheduleConfig config;
    config.numThread = m_thread_num;

    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)2; // 显式类型转换
    config.backendConfig = &backendConfig;

    NanoDet_session = NanoDet_interpreter->createSession(config);

    input_tensor = NanoDet_interpreter->getSessionInput(NanoDet_session, nullptr);

    std::cout << "=============================================================" << std::endl;
    std::cout << "DL Model infos and params" << std::endl;
    std::cout << "model: nanodet detection model for mnn" << std::endl;
    std::cout << "model weight: " << mnn_path << std::endl;
    std::cout << "model input width: " << m_input_width << std::endl;
    std::cout << "model input height: " << m_input_height << std::endl;
    std::cout << "number of threads: " << m_thread_num << std::endl;

    std::cout << "nms threshold: " << m_nms_threshold << std::endl;
    std::cout << "confidence threshold: " << m_confidence_threshold << std::endl;
    std::cout << "number of classes: " << m_class_num << std::endl;
    std::cout << "=============================================================" << std::endl;
}

NanoDetMNN::~NanoDetMNN()
{
    NanoDet_interpreter->releaseModel();
    NanoDet_interpreter->releaseSession(NanoDet_session);
}

std::shared_ptr<DetOutput> NanoDetMNN::Detect(const cv::Mat &img)
{
    std::shared_ptr<DetOutput> output(new DetOutput);
    if (img.empty()) {
        std::cout << "image is empty ,please check!" << std::endl;
        return output;
    }

    auto image_h = img.rows;
    auto image_w = img.cols;
    cv::Mat image;
    cv::resize(img, image, cv::Size(input_size[1], input_size[0]));

    NanoDet_interpreter->resizeTensor(input_tensor, { 1, 3, input_size[0], input_size[1] });
    NanoDet_interpreter->resizeSession(NanoDet_session);
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::BGR, mean_vals, 3, norm_vals, 3));
    pretreat->convert(image.data, input_size[1], input_size[0], image.step[0], input_tensor);

    auto start = std::chrono::steady_clock::now();

    // run network
    NanoDet_interpreter->runSession(NanoDet_session);

    // get output data
    std::vector<std::vector<BoxInfo>> results;
    results.resize(num_class);

    MNN::Tensor *tensor_preds = NanoDet_interpreter->getSessionOutput(NanoDet_session, output_name.c_str());

    MNN::Tensor tensor_preds_host(tensor_preds, tensor_preds->getDimensionType());
    tensor_preds->copyToHostTensor(&tensor_preds_host);

    // generate center priors in format of (x, y, stride)
    std::vector<CenterPrior> center_priors;
    generate_grid_center_priors(this->input_size[0], this->input_size[1], this->strides, center_priors);

    decode_infer(&tensor_preds_host, center_priors, m_confidence_threshold, results);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "inference time:" << elapsed.count() << " s, " << std::endl;

    // std::vector<BoxInfo> dets;
    for (int i = 0; i < (int)results.size(); i++) {
        nms(results[i], m_nms_threshold);

        for (auto box : results[i]) {
            box.x1 = box.x1 / input_size[1] * image_w;
            box.x2 = box.x2 / input_size[1] * image_w;
            box.y1 = box.y1 / input_size[0] * image_h;
            box.y2 = box.y2 / input_size[0] * image_h;
            m_result_list.push_back(box);

            DetResult det_result{ box.label, box.score, cv::Rect2f{ cv::Point2f{ box.x1, box.y1 }, cv::Point2f{ box.x2, box.y2 } } };
            output->m_res.emplace_back(det_result);
        }
    }
    std::cout << "detect " << output->m_res.size() << " objects" << std::endl;

    return output;
}

std::shared_ptr<DetModel> NanoDetMNN::GetModel(const std::string &config_path)
{
    std::shared_ptr<DetModel> model(new NanoDetMNN(config_path));
    if (!model->IsValid()) {
        return std::shared_ptr<DetModel>(nullptr);
    }
    return model;
}

void NanoDetMNN::decode_infer(MNN::Tensor *pred, std::vector<CenterPrior> &center_priors, float threshold, std::vector<std::vector<BoxInfo>> &results)
{
    const int num_points = center_priors.size();
    const int num_channels = num_class + (reg_max + 1) * 4;
    // printf("num_points:%d\n", num_points);

    // cv::Mat debug_heatmap = cv::Mat(feature_h, feature_w, CV_8UC3);
    for (int idx = 0; idx < num_points; idx++) {
        const int ct_x = center_priors[idx].x;
        const int ct_y = center_priors[idx].y;
        const int stride = center_priors[idx].stride;

        // preds is a tensor with shape [num_points, num_channels]
        const float *scores = pred->host<float>() + (idx * num_channels);

        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < num_class; label++) {
            if (scores[label] > score) {
                score = scores[label];
                cur_label = label;
            }
        }
        if (score > threshold) {
            const float *bbox_pred = pred->host<float>() + idx * num_channels + num_class;
            results[cur_label].push_back(disPred2Bbox(bbox_pred, cur_label, score, ct_x, ct_y, stride));
        }
    }
}

BoxInfo NanoDetMNN::disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y, int stride)
{
    float ct_x = x * stride;
    float ct_y = y * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++) {
        float dis = 0;
        float *dis_after_sm = new float[reg_max + 1];
        activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm, reg_max + 1);
        for (int j = 0; j < reg_max + 1; j++) {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        // std::cout << "dis:" << dis << std::endl;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], (float)input_size[1]);
    float ymax = (std::min)(ct_y + dis_pred[3], (float)input_size[0]);

    // std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    return BoxInfo{ xmin, ymin, xmax, ymax, score, label };
}

void NanoDetMNN::nms(std::vector<BoxInfo> &input_boxes, float nms_thresh)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= nms_thresh) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}