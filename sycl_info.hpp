#ifndef INFO_PRINT_HPP
#define INFO_PRINT_HPP
#include <CL/sycl.hpp>
#include <fstream>
void log_device_info(sycl::queue &q, const std::string& log_file)
{
    std::ofstream f(LOG_DIRECTORY + log_file);
    auto device = q.get_device();
    f << "name:\t" << device.template get_info<sycl::info::device::name>() << std::endl;
    f << "type:\t" << (uint32_t)device.template get_info<sycl::info::device::device_type>() << std::endl;
    f << "vendor:\t" << device.template get_info<sycl::info::device::vendor>() << std::endl;
    f << "version:\t" << device.template get_info<sycl::info::device::version>() << std::endl;
    f << "max_compute_units:\t" << device.template get_info<sycl::info::device::max_compute_units>() << std::endl;
    f << "max_work_group_size:\t" << device.template get_info<sycl::info::device::max_work_group_size>() << std::endl;
    f << "max_clock_frequency:\t" << device.template get_info<sycl::info::device::max_clock_frequency>() << std::endl;
    f << "global_mem_size:\t" << device.template get_info<sycl::info::device::global_mem_size>() << std::endl;
    f << "local_mem_size:\t" << device.template get_info<sycl::info::device::local_mem_size>() << std::endl;
    f << "max_mem_alloc_size:\t" << device.template get_info<sycl::info::device::max_mem_alloc_size>() << std::endl;
    f << "global_mem_cache_size:\t" << device.template get_info<sycl::info::device::global_mem_cache_size>() << std::endl;
    auto size_1D = device.template get_info<sycl::info::device::max_work_item_sizes<1>>();
    auto size_2D = device.template get_info<sycl::info::device::max_work_item_sizes<2>>();
    auto size_3D = device.template get_info<sycl::info::device::max_work_item_sizes<3>>();
    f << "max_work_item_sizes_1D:\t" << size_1D[0] << std::endl;
    f << "max_work_item_sizes_2D:\t" << size_2D[0] << size_2D[1] << std::endl;
    f << "max_work_item_sizes_3D:\t" << size_3D[0] << size_3D[1] << size_3D[2] << std::endl;
}

void log_total_info(sycl::queue &q_cpu, sycl::queue &q_gpu)
{
    const std::string str_cpu = "cpu.log";
    log_device_info(q_cpu, str_cpu);
    const std::string str_gpu = "gpu.log";
    log_device_info(q_gpu, str_gpu);
}

uint32_t get_work_group_size(sycl::queue &q)
{
    auto device = q.get_device();
    auto max_wg_size = device.template get_info<sycl::info::device::max_work_group_size>();
    return max_wg_size;
}

uint32_t get_max_compute_units(sycl::queue &q)
{
    auto device = q.get_device();
    auto max_compute_units = device.template get_info<sycl::info::device::max_compute_units>();
    return max_compute_units;
}

void log_kernel_infos(sycl::queue& q)
{

    auto ctx = q.get_context();
    auto device = q.get_device();
    auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx);
    auto k_ids = kb.get_kernel_ids();
    std::for_each(k_ids.begin(), k_ids.end(), [&](auto id)
    {
    std::ofstream f(LOG_DIRECTORY + std::string("kernel_id_") + id.get_name() + ".log");
    auto kernel = kb.get_kernel(id);
    f << "Kernel_id:\t" << id.get_name() << "\n";
    // f << "global_work_size:\t " << kernel.template get_info<sycl::info::kernel_device_specific::global_work_size>(device) << "\n";
    // f << "work_group_size:\t " << kernel.template get_info<sycl::info::kernel_device_specific::work_group_size>(device) << "\n";
    // f << "compile_work_group_size:\t " << kernel.template get_info<sycl::info::kernel_device_specific::compile_work_group_size>(device) << "\n";
    f << "preferred_work_group_size_multiple:\t " << kernel.template get_info<sycl::info::kernel_device_specific::preferred_work_group_size_multiple>(device) << "\n";
    f << "private_mem_size:\t " << kernel.template get_info<sycl::info::kernel_device_specific::private_mem_size>(device) << "\n";
    f << "max_num_sub_groups:\t " << kernel.template get_info<sycl::info::kernel_device_specific::max_num_sub_groups>(device) << "\n";
    f << "compile_num_sub_groups:\t " << kernel.template get_info<sycl::info::kernel_device_specific::compile_num_sub_groups>(device) << "\n";
    f << "max_sub_group_size:\t " << kernel.template get_info<sycl::info::kernel_device_specific::max_sub_group_size>(device) << "\n";
    f << "compile_sub_group_size:\t " << kernel.template get_info<sycl::info::kernel_device_specific::compile_sub_group_size>(device) << "\n";
    });
}

#endif
