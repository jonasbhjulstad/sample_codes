#ifndef BUFFER_CREATE_HPP
#define BUFFER_CREATE_HPP
#include <CL/sycl.hpp>
#include <algorithm>
#include <vector>
#include <cstdint>
// Buffer construction with explicit instantiation on device
template <typename T>
sycl::buffer<T, 1> buffer_create_1D(sycl::queue &q, const std::vector<T> &data, sycl::event &res_event)
{
    sycl::buffer<T> tmp(data.data(), data.size());
    sycl::buffer<T> result(sycl::range<1>(data.size()));

    res_event = q.submit([&](sycl::handler &h)
                         {
                auto tmp_acc = tmp.template get_access<sycl::access::mode::read>(h);
                auto res_acc = result.template get_access<sycl::access::mode::write>(h);
                h.copy(tmp_acc, res_acc); });
    return result;
}

template <typename T>
sycl::buffer<T, 2> buffer_create_2D(sycl::queue &q, const std::vector<T> &data, sycl::range<2> range, sycl::event &res_event)
{
    assert(data.size() == range[0]*range[1]);
    sycl::buffer<T,2> tmp(data.data(), range);
    sycl::buffer<T,2> result(range);

    res_event = q.submit([&](sycl::handler &h)
                         {
                auto tmp_acc = tmp.template get_access<sycl::access::mode::read>(h);
                auto res_acc = result.template get_access<sycl::access::mode::write>(h);
                h.copy(tmp_acc, res_acc); });
    return result;
}

template <typename T>
std::tuple<std::vector<sycl::buffer<T, 1>>, std::vector<sycl::event>> buffer_create_1D_vec(sycl::queue &q, const std::vector<T> &data, uint32_t N)
{
    std::vector<sycl::buffer<T, 1>> bufs;
    std::vector<sycl::event> events(N);
    bufs.reserve(N);
    std::transform(events.begin(), events.end(), std::back_inserter(bufs), [&](auto& event)
    {
        return buffer_create_1D(q, data, event);
    });
    return std::make_tuple(bufs, events);
}

#endif
