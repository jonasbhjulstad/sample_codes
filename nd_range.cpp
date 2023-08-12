#include <CL/sycl.hpp>
#include <cstdint>
#include <tuple>
#include "sycl_info.hpp"
#include "buffer_create.hpp"



sycl::event enqueue_nd_range(sycl::queue &q)
{
    auto N_wg = get_work_group_size(q);
    auto N_cu = get_max_compute_units(q);
    N_cu = (N_cu == 38) ? 32 : N_cu;
    // N_wg = 32;
    N_wg = 32;
    auto range = sycl::nd_range(sycl::range<1>(N_cu), sycl::range<1>(N_wg));
    std::vector<uint32_t> data(N_cu*N_wg, 2);
    uint32_t N_kernels = N_wg * N_cu;
    std::vector<sycl::event> init_events(2);
    auto read_buf = buffer_create_2D(q, data, sycl::range<2>(N_cu, N_wg), init_events[0]);
    auto result_buf = buffer_create_2D(q, data, sycl::range<2>(N_cu, N_wg), init_events[1]);

    return q.submit([&](sycl::handler& h)
    {
        auto read_acc = read_buf.template get_access<sycl::access::mode::read>(h);
        auto res_acc = result_buf.template get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(range, [=](sycl::nd_item<1> it)
        {
            auto gid = it.get_global_id();
            auto lid = it.get_local_id();
            res_acc[gid][lid] += read_acc[gid][lid];
        });
    });

}


int main()
{

    sycl::queue q_cpu(sycl::cpu_selector_v);
    sycl::queue q_gpu(sycl::gpu_selector_v);

    log_total_info(q_cpu, q_gpu);
    log_kernel_infos(q_cpu);
    log_kernel_infos(q_gpu);


    auto cpu_event = enqueue_nd_range(q_cpu);
    auto gpu_event = enqueue_nd_range(q_gpu);

    cpu_event.wait();
    gpu_event.wait();
    return 0;
}
