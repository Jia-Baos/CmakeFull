#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

#include <opencv2/opencv.hpp>

std::atomic<bool> event_flag(false);

std::mutex mtx;
bool ready = false;
std::condition_variable cond;
std::function<void()> callback;

void WorkerThread()
{
    while (true) {
        {
            std::unique_lock<std::mutex> lck(mtx);
            cond.wait(lck, [] { return event_flag.load() || ready; }); // 等待事件或条件成立
            std::cout << "Worker thread waiting..." << std::endl;
        }
        if (event_flag.exchange(false)) { // 如果事件被设置，则清除并执行回调
            callback();                   // 执行回调函数
            std::cout << "Callback executed in thread!" << std::endl;
        } else if (ready) { // 如果条件成立，则退出循环（可选）
            break;
        }
    }
}

void SetEvent()
{
    event_flag.store(true); // 设置事件标志位为true
    cond.notify_one();      // 通知等待的线程
    std::cout << "Event set!" << std::endl;
}

int main()
{
    std::thread t(WorkerThread);
    {
        std::lock_guard<std::mutex> lck(mtx);
        callback = []() { std::cout << "Callback executed!" << std::endl; };
        // ready = true; // 设置条件成立标志位（可选），设置线程可以退出
    }

    // cv.notify_one(); // 通知等待的线程（可选）
    SetEvent(); // 设置事件，触发回调执行（可选）
    std::this_thread::sleep_for(std::chrono::seconds(1));
    SetEvent(); // 设置事件，触发回调执行（可选）

    {
        std::lock_guard<std::mutex> lck(mtx);
        ready = true; // 设置条件成立标志位（可选），设置线程可以退出
    }

    t.join();

    std::string filename = "/home/jia-baos/Project-Cpp/CmakeFull/config/output.yaml";
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    fs << "camera_name" << "cam-000";
    fs << "image_width" << 640;
    fs << "image_height" << 480;

    fs << "distortion_parameters";
    fs << "{"
       << "k1" << 0 << "k2" << 0 << "p1" << 0 << "p2" << 0 << "}";

    fs.release();

    cv::FileStorage fs_read(filename, cv::FileStorage::READ);

    std::string camera_name;
    fs_read["camera_name"] >> camera_name;
    std::cout << camera_name << std::endl;
    auto image_width = static_cast<int>(fs_read["image_width"]);
    auto image_height = static_cast<int>(fs_read["image_height"]);
    std::cout << "image_width: " << image_width << std::endl;
    std::cout << "image_height: " << image_height << std::endl;

    cv::FileNode distortion_parameters = fs_read["distortion_parameters"];
    auto k1 = static_cast<double>(distortion_parameters["k1"]);
    auto k2 = static_cast<double>(distortion_parameters["k2"]);
    auto p1 = static_cast<double>(distortion_parameters["p1"]);
    auto p2 = static_cast<double>(distortion_parameters["p2"]);
    std::cout << "k1: " << k1 << std::endl;
    std::cout << "k2: " << k2 << std::endl;
    std::cout << "p1: " << p1 << std::endl;
    std::cout << "p2: " << p2 << std::endl;

    return 0;
}
