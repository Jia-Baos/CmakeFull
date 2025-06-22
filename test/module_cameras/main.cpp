#include <thread>
#include <opencv2/opencv.hpp>
#include <camera_orbbec.hpp>

int main()
{
    OrbbecCamera cam;
    cam.SetSN("1234567890");

    std::thread cam_thread = std::thread([&cam]() {
        cam.Run();
    });
    cam_thread.detach();

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        cv::Mat img{};
        auto res = cam.GetImg();
        if (res.has_value()) {
            img = res.value();
            std::cout << "get color img" << std::endl;
        } else {
            std::cout << "get color img failed" << std::endl;
        }
    }

    return 0;
}
