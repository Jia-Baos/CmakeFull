#define FMT_HEADER_ONLY
#include <safe_queue.hpp>
#include <task_queue.hpp>

// #include <fmt/core.h>
// #include <fmt/chrono.h>
// #include <fmt/color.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

int main(int argc, char *argv[])
{
    // fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "This is red and bold\n");

    // TaskQueue<int> task_queue;
    // int val{ 1 };
    // task_queue.Push(1);
    // task_queue.Push(2);
    // task_queue.Push(3);

    // fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "task_queue size {}\n", task_queue.Size());
    // fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "task_queue is empty {}\n", task_queue.Empty());

    // task_queue.TryPop(val);
    // fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "task_queue.TryPop, get value {}\n", val);
    // task_queue.WaitAndPop(val);
    // fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "task_queue.WaitAndPop, get value {}\n", val);

    // task_queue.Clear();
    // fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "task_queue size {}\n", task_queue.Size());
    // fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "task_queue is empty {}\n", task_queue.Empty());

    // val = 0;
    // task_queue.StopSignal(val, 3);
    // fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "task_queue size {}\n", task_queue.Size());
    // fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "task_queue is empty {}\n", task_queue.Empty());

    // 创建文件日志器（追加模式，默认）
    auto file_logger = spdlog::basic_logger_mt("file_logger", "logs/my_log.txt");

    // 设置日志级别（可选，默认info）
    file_logger->set_level(spdlog::level::info);

    // 写入日志
    file_logger->info("Hello, this is an info message!");
    file_logger->error("This is an error message with number: {}", 42);

    // 刷新缓冲区（重要！确保日志写入磁盘）
    file_logger->flush();

    return 0;
}
