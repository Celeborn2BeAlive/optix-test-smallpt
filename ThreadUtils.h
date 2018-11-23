#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace shn
{

// \return The number of logical threads for the system
inline uint32_t getSystemThreadCount()
{
    return std::thread::hardware_concurrency();
}

// Run a functor asynchronously in multiple threads.
//
// \arg threadCount Number of threads to start
// \arg task Functor to execute
//
// \return A future that will be set when the last thread finishes. Allows synchronization by waiting on it.
template<typename TaskFunctor>
inline std::future<void> asyncParallelRun(uint32_t threadCount, TaskFunctor task)
{
    struct SharedData
    {
        std::promise<void> p;
        ~SharedData()
        {
            p.set_value();
        }
    };
    std::shared_ptr<SharedData> sharedData = std::make_shared<SharedData>();
    for (auto i = 0u; i < threadCount; ++i)
    {
        std::thread t([i, task, sharedData]() { task(i); });
        t.detach();
    }
    return sharedData->p.get_future();
}

// Run a functor asynchronously in multiple threads.
//
// \arg threadCount Number of threads to start
// \arg task Functor to execute
// \arg completeCallback A callback that will be called by the last thread when it finishes.
//
// \return A future that will be set when the last thread finishes. Allows synchronization by waiting on it.
template<typename TaskFunctor, typename CompleteFunctor>
inline std::future<void> asyncParallelRun(uint32_t threadCount, TaskFunctor task, CompleteFunctor completeCallback)
{
    struct SharedData
    {
        CompleteFunctor completeCallback;
        SharedData(CompleteFunctor completeCallback): completeCallback(completeCallback)
        {
        }
        ~SharedData()
        {
            completeCallback();
        }
    };

    std::shared_ptr<SharedData> sharedData = std::make_shared<SharedData>(completeCallback);
    return asyncParallelRun(threadCount, [task, sharedData](size_t threadId) { task(threadId); });
}

// \brief Run in parallel a task and wait for it to finish.
//
// \tparam ThreadStackData A class that will be instanciated by each thread before running the task. The instance will be passed to the task. It must be default constructible.
// \tparam TaskFunctor The task functor that will be executed. Prototype must be (void)(size_t runId, size_t threadId, ThreadStackData & data)
// \tparam InitThreadStackDataFunctor A functor to initialize the stack data. Prototype must be (void)(ThreadStackData & data)
//
// \arg runCount Number of time the task should be executed
// \arg threadCount Number of threads to be used (not counting the calling thread, that will wait others to finish their tasks)
// \arg task The task to execute
// \arg init The functor to initialize the stack data
//
template<typename ThreadStackData, typename TaskFunctor, typename InitThreadStackDataFunctor>
inline std::future<void> asyncParallelLoop(uint32_t runCount, uint32_t threadCount, TaskFunctor task, InitThreadStackDataFunctor init)
{
    const auto blockSize = std::max(1u, runCount / threadCount);

    std::shared_ptr<std::atomic_uint> nextBatch = std::make_shared<std::atomic_uint>(0);
    auto batchProcess = [blockSize, nextBatch, runCount, task, init](uint32_t threadID) {
        ThreadStackData threadData;
        init(threadData); // Here I would have prefer to use constructor with variadic template arguments but GCC has this bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=47226
        while (true)
        {
            auto batchID = (*nextBatch)++;
            auto taskID = batchID * blockSize;

            if (taskID >= runCount)
            {
                break;
            }

            auto end = std::min(taskID + blockSize, runCount);

            while (taskID < end)
            {
                task(taskID, threadID, threadData);
                ++taskID;
            }
        }
    };

    return asyncParallelRun(threadCount, batchProcess);
}

// Specialization with an empty initialization functor for stack data
template<typename ThreadStackData, typename TaskFunctor>
inline std::future<void> asyncParallelLoop(uint32_t runCount, uint32_t threadCount, TaskFunctor task)
{
    return asyncParallelLoop<ThreadStackData>(runCount, threadCount, task, [&](ThreadStackData &) {});
}

// Specialization for no stack data
template<typename TaskFunctor>
inline std::future<void> asyncParallelLoop(uint32_t runCount, uint32_t threadCount, TaskFunctor task)
{
    struct NullStruct
    {
    };
    return asyncParallelLoop<NullStruct>(runCount, threadCount, [&](size_t taskId, size_t threadId, NullStruct &) { task(taskId, threadId); });
}

template<typename TaskFunctor>
inline void syncParallelRun(uint32_t threadCount, TaskFunctor task)
{
    asyncParallelRun(threadCount, task).wait();
}

template<typename TaskFunctor, typename CompleteFunctor>
inline void syncParallelRun(uint32_t threadCount, TaskFunctor task, CompleteFunctor completeCallback)
{
    asyncParallelRun(threadCount, task, completeCallback).wait();
}

template<typename ThreadStackData, typename TaskFunctor, typename InitThreadStackDataFunctor>
inline void syncParallelLoop(uint32_t runCount, uint32_t threadCount, TaskFunctor task, InitThreadStackDataFunctor init)
{
    asyncParallelLoop<ThreadStackData>(runCount, threadCount, task, init).wait();
}

template<typename ThreadStackData, typename TaskFunctor>
inline void syncParallelLoop(uint32_t runCount, uint32_t threadCount, TaskFunctor task)
{
    asyncParallelLoop<ThreadStackData>(runCount, threadCount, task).wait();
}

template<typename TaskFunctor>
inline void syncParallelLoop(uint32_t runCount, uint32_t threadCount, TaskFunctor task)
{
    asyncParallelLoop(runCount, threadCount, task).wait();
}

} // namespace shn
