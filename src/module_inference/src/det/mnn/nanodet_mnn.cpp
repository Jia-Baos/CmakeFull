#include "../../../include/det/mnn/nanodet_mnn.hpp"
#include <iostream>
#include <chrono>

inline std::string replaceExtension(std::string input, std::string new_ext)
{
    return input.substr(0, input.find_last_of('.')) + new_ext;
}

inline void generate_grid_center_priors(const int input_height, const int input_width, const std::vector<int> &strides, std::vector<NanoDet::CenterPrior> &center_priors)
{
    for (int i = 0; i < (int)strides.size(); i++) {
        int stride = strides[i];
        int feat_w = std::ceil((float)input_width / stride);
        int feat_h = std::ceil((float)input_height / stride);
        for (int y = 0; y < feat_h; y++) {
            for (int x = 0; x < feat_w; x++) {
                NanoDet::CenterPrior ct;
                ct.x = x;
                ct.y = y;
                ct.stride = stride;
                center_priors.push_back(ct);
            }
        }
    }
}

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

template <typename _Tp>
inline int activation_function_softmax(const _Tp *src, _Tp *dst, int length)
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

static cv::Mat DrawBoxes(const cv::Mat &img, const std::vector<NanoDet::BoxInfo> &bboxes, const ObjectRect effect_roi)
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

    MNN::BackendConfig backend_config;
    backend_config.precision = (MNN::BackendConfig::PrecisionMode)2; // 显式类型转换
    config.backendConfig = &backend_config;

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

    cv::Mat image;
    const int kWidth = this->GetInputWidth();
    const int kHeight = this->GetInputHeight();
    ResizeUniform(img, image, cv::Size(kWidth, kHeight), this->m_object_rect); // -> 416*416
    std::cout << "ResizeUniform(), src size: " << image.size() << std::endl;

    NanoDet_interpreter->resizeTensor(input_tensor, { 1, 3, this->m_input_height, this->m_input_width });
    NanoDet_interpreter->resizeSession(NanoDet_session);
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::BGR, mean_vals, 3, norm_vals, 3));
    pretreat->convert(image.data, this->m_input_width, this->m_input_height, image.step[0], input_tensor);

    auto start = std::chrono::steady_clock::now();

    // run network
    NanoDet_interpreter->runSession(NanoDet_session);

    // get output data
    std::vector<std::vector<NanoDet::BoxInfo>> results;
    results.resize(num_class);

    MNN::Tensor *tensor_preds = NanoDet_interpreter->getSessionOutput(NanoDet_session, output_name.c_str());

    MNN::Tensor tensor_preds_host(tensor_preds, tensor_preds->getDimensionType());
    tensor_preds->copyToHostTensor(&tensor_preds_host);

    // generate center priors in format of (x, y, stride)
    std::vector<NanoDet::CenterPrior> center_priors;
    generate_grid_center_priors(this->m_input_height, this->m_input_width, this->strides, center_priors);

    decode_infer(tensor_preds_host, center_priors, m_confidence_threshold, results);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "inference time:" << elapsed.count() << " s, " << std::endl;

    // std::vector<BoxInfo> dets;
    m_result_list.clear();
    for (int i = 0; i < (int)results.size(); i++) {
        nms(results[i], m_nms_threshold);

        for (auto box : results[i]) {
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

void NanoDetMNN::decode_infer(const MNN::Tensor &pred, const std::vector<NanoDet::CenterPrior> &center_priors, const float threshold, std::vector<std::vector<NanoDet::BoxInfo>> &results)
{
    const int kNumPoints = center_priors.size();
    const int kNumChannels = num_class + (reg_max + 1) * 4;
    // printf("num_points:%d\n", num_points);

    // cv::Mat debug_heatmap = cv::Mat(feature_h, feature_w, CV_8UC3);
    for (int idx = 0; idx < kNumPoints; idx++) {
        const int kCtX = center_priors[idx].x;
        const int kCtY = center_priors[idx].y;
        const int kStride = center_priors[idx].stride;

        // preds is a tensor with shape [num_points, num_channels]
        const float *scores = pred.host<float>() + (idx * kNumChannels);

        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < num_class; label++) {
            if (scores[label] > score) {
                score = scores[label];
                cur_label = label;
            }
        }
        if (score > threshold) {
            const float *bbox_pred = pred.host<float>() + idx * kNumChannels + num_class;
            results[cur_label].push_back(disPred2Bbox(bbox_pred, cur_label, score, kCtX, kCtY, kStride));
        }
    }
}

NanoDet::BoxInfo NanoDetMNN::disPred2Bbox(const float *dfl_det, const int label, const float score, const int x, const int y, const int stride)
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

    // https://zhuanlan.zhihu.com/p/452602582
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], (float)this->m_input_width);
    float ymax = (std::min)(ct_y + dis_pred[3], (float)this->m_input_height);

    // std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    return NanoDet::BoxInfo{ xmin, ymin, xmax, ymax, score, label };
}

void NanoDetMNN::nms(std::vector<NanoDet::BoxInfo> &input_boxes, const float nms_thresh)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](NanoDet::BoxInfo a, NanoDet::BoxInfo b) { return a.score > b.score; });
    std::vector<float> v_area(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        v_area[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
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
            float ovr = inter / (v_area[i] + v_area[j] - inter);
            if (ovr >= nms_thresh) {
                input_boxes.erase(input_boxes.begin() + j);
                v_area.erase(v_area.begin() + j);
            } else {
                j++;
            }
        }
    }
}
