#include <thread>
#include <opencv2/opencv.hpp>
#include "camera_orbbec.hpp"

int main()
{
    OrbbecCamera cam;
    cam.SetSN("AD74B3300X2");
    cam.StartCapture();

    // std::thread cam_thread = std::thread([&cam]() {
    //     cam.Run();
    // });
    // cam_thread.detach();

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        cv::Mat img{};
        auto res = cam.GetDataFrame();
        if (res.has_value() && !res.value().img.empty()) {
            img = res.value().img;
            // std::cout << "get color img" << std::endl;
        } else {
            std::cout << "get color img failed" << std::endl;
        }

        if (img.empty()) {
            continue;
        }
        
        cv::imshow("img", img);
        // 必须加 waitKey，否则窗口不刷新
        if (cv::waitKey(1) == 27) { // 按ESC退出
            break;
        }
    }

    return 0;
}
