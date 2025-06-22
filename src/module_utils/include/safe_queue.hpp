#ifndef MODULE_UTILS_SAFE_QUEUE_HPP
#define MODULE_UTILS_SAFE_QUEUE_HPP

#include <mutex>
#include <queue>
#include <algorithm>
#include <condition_variable>

template <typename T>
class SafeQueue {
public:
    SafeQueue() = default;
    ~SafeQueue() = default;

    SafeQueue(const SafeQueue &other) = delete;
    SafeQueue &operator=(const SafeQueue &other) = delete;

    SafeQueue(SafeQueue &&other) = delete;
    SafeQueue &operator=(SafeQueue &&other) = delete;

    bool Empty()
    {
        std::unique_lock<std::mutex> locker(m_mutex);
        return m_queue.empty();
    }

    int Size()
    {
        std::unique_lock<std::mutex> locker(m_mutex);
        return m_queue.size();
    }

    void Clear()
    {
        std::unique_lock<std::mutex> locker(m_mutex);

        std::queue<T> t_safe_queue{};
        std::swap(m_queue, t_safe_queue);
    }

    // 向队列中推入一个元素（左值版本）。
    // @param val 需要推入队列的元素（左值引用）
    void Push(T &val)
    {
        std::unique_lock<std::mutex> locker(m_mutex);
        m_queue.emplace(val);
        m_cond.notify_one();
    }

    // 向队列中推入一个元素（右值版本，支持移动语义）。
    // @param val 需要推入队列的元素（右值引用）
    void Push(T &&val)
    {
        std::unique_lock<std::mutex> locker(m_mutex);
        m_queue.emplace(std::move(val));
        m_cond.notify_one();
    }

    // 尝试从队列中弹出一个元素（非阻塞）。
    // @param val 用于接收弹出的元素
    // @return    如果成功弹出元素返回true，队列为空返回false
    bool TryPop(T &val)
    {
        std::unique_lock<std::mutex> locker(m_mutex);
        if (m_queue.empty())
            return false;

        val = std::move(m_queue.front());
        m_queue.pop();
        return true;
    }

    // 阻塞等待直到队列非空，并弹出一个元素。
    // @param val 用于接收弹出的元素
    // @return    总是返回true（因为一定能弹出元素）
    bool WaitAndPop(T &val)
    {
        std::unique_lock<std::mutex> locker(m_mutex);
        // 调用wait()前，此时线程应该是加锁状态，因为wait()在阻塞线程时，会自动调用lck.unlock()释放锁，使得其他被阻塞在锁竞争上的线程得以执行
        // 当前线程被另一线程notify_ *() 唤醒时，则会自动调用lck.lock() 加锁，使得lck状态和wait() 被调用时相同
        m_cond.wait(locker, [this] { return !m_queue.empty(); });

        val = std::move(m_queue.front());
        m_queue.pop();
        return true;
    }

    // 在指定超时时间内等待队列非空，并弹出一个元素。
    // @param val     用于接收弹出的元素
    // @param timeout 最长等待时间（std::chrono::milliseconds）
    // @return        成功弹出返回true，超时未弹出返回false
    bool WaitForAndPop(T &val, std::chrono::milliseconds timeout)
    {
        std::unique_lock<std::mutex> locker(m_mutex);
        if (!m_cond.wait_for(locker, timeout, [this] { return !m_queue.empty(); }))
            return false;

        val = std::move(m_queue.front());
        m_queue.pop();
        return true;
    }

    // 等待直到队列非空或超时，从队列中取出一个元素并移除。
    // @param val     用于接收弹出的元素
    // @param timeout 等待的绝对超时时间点（std::chrono::system_clock）
    // @return        如果成功取出元素返回true，超时未取到返回false
    bool WaitUntilAndPop(T &val, std::chrono::time_point<std::chrono::system_clock> timeout)
    {
        std::unique_lock<std::mutex> locker(m_mutex);
        if (!m_cond.wait_until(locker, timeout, [this] { return !m_queue.empty(); }))
            return false;

        val = std::move(m_queue.front());
        m_queue.pop();
        return true;
    }

protected:
    std::mutex m_mutex;
    std::queue<T> m_queue;
    std::condition_variable m_cond;
};
#endif // MODULE_UTILS_SAFE_QUEUE_HPP
