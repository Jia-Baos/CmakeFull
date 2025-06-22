#define FMT_HEADER_ONLY
#include <safe_queue.hpp>
#include <task_queue.hpp>

#include <fmt/core.h>
#include <fmt/chrono.h>
#include <fmt/color.h>

int main(int argc, char *argv[])
{
    fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "This is red and bold\n");

    TaskQueue<int> task_queue;
    int val{ 1 };
    task_queue.Push(1);
    task_queue.Push(2);
    task_queue.Push(3);

    fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "task_queue size {}\n", task_queue.Size());
    fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "task_queue is empty {}\n", task_queue.Empty());

    task_queue.TryPop(val);
    fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "task_queue.TryPop, get value {}\n", val);
    task_queue.WaitAndPop(val);
    fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "task_queue.WaitAndPop, get value {}\n", val);

    task_queue.Clear();
    fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "task_queue size {}\n", task_queue.Size());
    fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "task_queue is empty {}\n", task_queue.Empty());

    val = 0;
    task_queue.StopSignal(val, 3);
    fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "task_queue size {}\n", task_queue.Size());
    fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "task_queue is empty {}\n", task_queue.Empty());
    return 0;
}
