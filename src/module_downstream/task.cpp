#include "./task.hpp"

#include <iostream>
#include <typeinfo>
#include <opencv2/opencv.hpp>

static cv::Mat DrawBoxes(const cv::Mat &img, const std::vector<NanoDet::BoxInfo> &bboxes, ObjectRect effect_roi)
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

TaskManager::TaskManager()
{
    std::cout << "Init TaskManager" << std::endl;
}

TaskManager::~TaskManager()
{
    std::cout << "Destory TaskManager" << std::endl;
}

bool TaskManager::Init()
{
    std::cout << "TaskManager::Init() in" << std::endl;

    std::string config_path = "/home/jia-baos/Project-Cpp/CmakeFull/config/nanodet_mnn.yaml";
    m_model = DLModel::GetModel(config_path);

    std::cout << "TaskManager::Init() out" << std::endl;

    return true;
}

bool TaskManager::Run()
{
    std::cout << "TaskManager::Run() in" << std::endl;

    this->Start();

    std::cout << "TaskManager::Run() out" << std::endl;

    return true;
}

void TaskManager::Start()
{
    std::cout << "TaskManager::Start() in" << std::endl;

    if (!m_stop_flag) {

        m_infer_task.Clear();
        m_stop_flag.store(false);

        m_get_img_run_thread = std::make_shared<std::thread>(&TaskManager::GetImgRun, this);
        m_dl_infer_run_thread = std::make_shared<std::thread>(&TaskManager::DLInferRun, this);
        m_post_processing_run_thread = std::make_shared<std::thread>(&TaskManager::PostProcessingRun, this);
    }

    std::cout << "TaskManager::Start() out" << std::endl;
}

void TaskManager::Stop()
{
    std::cout << "TaskManager::Stop() in" << std::endl;

    if (!m_stop_flag) {
        m_stop_flag = true;

        if (m_get_img_run_thread) {
            m_get_img_run_thread->join();

            InferTaskData infer_task_data;
            infer_task_data.raw_data_frame = nullptr;
            m_infer_task.StopSignal(infer_task_data, 1);
            m_dl_infer_run_thread->join();
        }

        if (m_post_processing_run_thread) {
            PostProcessingData post_processing_data;
            post_processing_data.raw_data_frame = nullptr;
            m_post_processing_task.StopSignal(post_processing_data, 1);
            m_post_processing_run_thread->join();
        }
    }

    std::cout << "TaskManager::Stop() out" << std::endl;
}

void TaskManager::GetImgRun()
{
    std::cout << "TaskManager::GetImgRun() in" << std::endl;

    while (!m_stop_flag) {
        // std::shared_ptr<Frame> frame = m_camera->getImage();

        const std::string img_path = "/home/jia-baos/Project-Cpp/CmakeFull/imgs/000252.jpg";
        const cv::Mat img = cv::imread(img_path);

        // 模拟采图延迟，后面所有处理的耗时一定要小于此处采图的耗时，否则数据会一直在 m_infer_task 中堆积导致后面处理中判断超时
        std::this_thread::sleep_for(std::chrono::milliseconds(150));

        const uint64_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        // std::cout << "TaskManager::GetImgRun(), timestamp: " << timestamp << std::endl;

        std::shared_ptr<DataFrame> data_frame = std::make_shared<DataFrame>();
        data_frame->img = img;
        data_frame->timestamp = timestamp;

        if (!data_frame) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        InferTaskData infer_task_data{};
        infer_task_data.img = img;
        infer_task_data.raw_data_frame = data_frame;
        m_infer_task.Push(infer_task_data);
    }

    std::cout << "TaskManager::GetImgRun() out" << std::endl;
}

void TaskManager::DLInferRun()
{
    std::cout << "TaskManager::DLInferRun() in" << std::endl;

    while (!m_stop_flag) {
        InferTaskData infer_task_data{};
        m_infer_task.WaitAndPop(infer_task_data);

        if (!infer_task_data.raw_data_frame) {
            continue;
        }

        const uint64_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        std::cout << "TaskManager::DLInferRun(), raw_data_frame->timestamp: " << infer_task_data.raw_data_frame->timestamp << std::endl;
        std::cout << "TaskManager::DLInferRun(),                 timestamp: " << timestamp << std::endl;
        
        const double delay_time = (timestamp - infer_task_data.raw_data_frame->timestamp) / 1000.0;
        if (delay_time > m_delay_time) {
            std::cout << "TaskManager::DLInferRun() deprive old data" << std::endl;
            continue;
        }

        std::shared_ptr<DLOutput> out = m_model->Infer(infer_task_data.img);

        // 需要根据任务类型手动转换到子类，否则无法获取结果
        if (out->m_type == ModelType::kDetection) {
            auto derivate_ptr = std::dynamic_pointer_cast<DetOutput>(out);
            std::cout << "ModelType: " << ModelTypeToString(derivate_ptr->m_type) << std::endl;
            // std::cout << "timestamp: " << infer_task_data.raw_data_frame->timestamp << std::endl;
        }

        auto derivate_ptr = std::dynamic_pointer_cast<NanoDetMNN>(m_model);
        auto object_rect = derivate_ptr->m_object_rect;
        auto results = derivate_ptr->m_result_list;

        if (results.empty()) {
            std::cout << "TaskManager::DLInferRun() results is empty" << std::endl;
            continue;
        }

        PostProcessingData post_processing_data{};
        post_processing_data.object_rect = object_rect;
        post_processing_data.box_info = results;
        post_processing_data.raw_data_frame = infer_task_data.raw_data_frame;
        m_post_processing_task.Push(post_processing_data);
    }

    std::cout << "TaskManager::DLInferRun() out" << std::endl;
}

void TaskManager::PostProcessingRun()
{
    std::cout << "TaskManager::PostProcessingRun() in" << std::endl;

    while (!m_stop_flag) {
        PostProcessingData post_processing_data{};
        m_post_processing_task.WaitAndPop(post_processing_data);

        if (!post_processing_data.raw_data_frame) {
            continue;
        }

        const uint64_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        std::cout << "TaskManager::PostProcessingRun(), raw_data_frame->timestamp: " << post_processing_data.raw_data_frame->timestamp << std::endl;
        std::cout << "TaskManager::PostProcessingRun(),                 timestamp: " << timestamp << std::endl;
        
        const double delay_time = (timestamp - post_processing_data.raw_data_frame->timestamp) / 1000.0;
        if (delay_time > m_delay_time) {
            std::cout << "TaskManager::PostProcessingRun() deprive old data" << std::endl;
            continue;
        }

        cv::Mat res = DrawBoxes(post_processing_data.raw_data_frame->img, post_processing_data.box_info, post_processing_data.object_rect);

        std::string save_path = "/home/jia-baos/Project-Cpp/CmakeFull/install/res/"
                                + std::to_string(post_processing_data.raw_data_frame->timestamp) + ".png";
        cv::imwrite(save_path, res);
    }

    std::cout << "TaskManager::PostProcessingRun() out" << std::endl;
}

int main()
{
    std::shared_ptr<TaskManager> task_manager = std::make_shared<TaskManager>();
    task_manager->Init();
    task_manager->Run();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    task_manager->Stop();

    return 0;
}
