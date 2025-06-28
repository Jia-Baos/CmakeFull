#pragma once
#include <iostream>
#include <atomic>
#include <memory>
#include <thread>
#include <opencv2/opencv.hpp>

#include "task_queue.hpp"

#include "model_base.hpp"
#include "model_det.hpp"
#include "nanodet_mnn.hpp"

struct DataFrame {
    cv::Mat img;
    uint64_t timestamp;
};

struct InferTaskData {
    cv::Mat img;
    std::shared_ptr<DataFrame> raw_data_frame; // raw data
};

struct PostProcessingData {
    ObjectRect object_rect;
    std::vector<NanoDet::BoxInfo> box_info;
    std::shared_ptr<DataFrame> raw_data_frame; // raw data
};

class TaskManager {
public:
    TaskManager();
    ~TaskManager();

    bool Init();
    bool Run();
    void Start();
    void Stop();

private:
    void GetImgRun();
    void DLInferRun();
    void PostProcessingRun();

private:
    std::shared_ptr<std::thread> m_get_img_run_thread;
    std::shared_ptr<std::thread> m_dl_infer_run_thread;
    std::shared_ptr<std::thread> m_post_processing_run_thread;

    std::shared_ptr<DLModel> m_model;

    int m_delay_time = 200; // ms
    std::atomic<bool> m_stop_flag{ false };
    TaskQueue<InferTaskData> m_infer_task;
    TaskQueue<PostProcessingData> m_post_processing_task;
};