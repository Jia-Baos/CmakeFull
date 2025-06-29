#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>
// #include <spdlog/fmt/ostr.h>
#include <spdlog/fmt/bundled/color.h>
#include <spdlog/sinks/basic_file_sink.h>

// spdlog 4.x 及以上自带 fmt

int main(int argc, char *argv[])
{
    fmt::print(fmt::fg(fmt::color::red) | fmt::emphasis::bold, "This is red and bold\n");

    // 创建文件日志器（追加模式，默认）
    auto file_logger = spdlog::basic_logger_mt("file_logger", "Log/my_log.txt");

    // 设置日志级别（可选，默认info）
    file_logger->set_level(spdlog::level::info);

    // 写入日志
    file_logger->info("Hello, this is an info message!");
    file_logger->error("This is an error message with number: {}", 42);

    // 刷新缓冲区（重要！确保日志写入磁盘）
    file_logger->flush();

    return 0;
}
