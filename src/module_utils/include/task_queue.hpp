#ifndef MODULE_UTILS_TASK_QUEUE_HPP
#define MODULE_UTILS_TASK_QUEUE_HPP

#include "safe_queue.hpp"
#include <cstddef>

template <typename T>
class TaskQueue : public SafeQueue<T> {
public:
    void StopSignal(T &val, const int n)
    {
        std::unique_lock<std::mutex> locker(this->m_mutex);
        for (size_t i = 0; i < n; i++) {
            this->m_queue.emplace(val);
        }
        this->m_cond.notify_all();
    }
};

#endif // MODULE_UTILS_TASK_QUEUE_HPP
