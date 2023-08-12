#include <CL/sycl.hpp>
#include <cstdint>
#include <tuple>
#include "sycl_info.hpp"
#include "buffer_create.hpp"


std::vector<sycl::event> enqueue_separate_kernels(sycl::queue &q)
{
    auto N_wg = get_work_group_size(q);
    auto N_cu = get_max_compute_units(q);
    std::vector<uint32_t> data(N_wg, 2);
    uint32_t N_kernels = N_cu;

    auto [read_bufs, init_read_events] = buffer_create_1D_vec(q, data, N_cu);
    auto [result_bufs, init_result_events] = buffer_create_1D_vec(q, data, N_cu);

    auto kernel_enqueue = [&](auto &b0, auto &b1, auto &dep_read_event, auto& dep_result_event)
    { return
          [&](sycl::handler &h)
      {
          h.depends_on(dep_read_event);
          h.depends_on(dep_result_event);
          auto read_acc = b0.template get_access<sycl::access::mode::read>(h);
          auto res_acc = b1.template get_access<sycl::access::mode::read_write>(h);
          h.parallel_for(N_wg, [=](sycl::id<1> id)
                         { res_acc[id] += read_acc[id]; });
      }; };

    std::vector<sycl::event> res_events(N_kernels);
    for (int i = 0; i < N_kernels; i++)
    {
        res_events[i] = q.submit(kernel_enqueue(read_bufs[i], result_bufs[i], init_read_events[i], init_result_events[i]));
    }
    return res_events;
}


int main()
{

    sycl::queue q_cpu(sycl::cpu_selector_v);
    sycl::queue q_gpu(sycl::gpu_selector_v);

    log_total_info(q_cpu, q_gpu);
    log_kernel_infos(q_cpu);
    log_kernel_infos(q_gpu);


    auto cpu_events = enqueue_separate_kernels(q_cpu);
    auto gpu_events = enqueue_separate_kernels(q_gpu);

    std::for_each(cpu_events.begin(), cpu_events.end(), [&](auto &e)
                  { e.wait(); });
    std::for_each(gpu_events.begin(), gpu_events.end(), [&](auto &e)
                  { e.wait(); });

    return 0;
}
