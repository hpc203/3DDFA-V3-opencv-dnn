# ifndef RENDER
# define RENDER
#include <opencv2/core.hpp>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdint>


class MeshRenderer_cpu
{
public:
    MeshRenderer_cpu(const float rasterize_fov, const float znear=0.1, const float zfar=10, const int rasterize_size=224);
    std::tuple<cv::Mat, cv::Mat, cv::Mat, std::vector<int>> forward(const cv::Mat& vertex, const int* tri, const std::vector<int> tri_shape, const cv::Mat& feat=cv::Mat(), const bool visible_vertice=false);
private:
    cv::Mat ndc_proj;
    int rasterize_size;
};

class MeshRenderer_UV_cpu
{
public:
    MeshRenderer_UV_cpu(const int rasterize_size=224);
    std::tuple<cv::Mat, cv::Mat, cv::Mat, std::vector<int>> forward(const cv::Mat& vertex, const int* tri, const std::vector<int> tri_shape, const cv::Mat& feat=cv::Mat(), const bool visible_vertice=false);
private:
    int rasterize_size;
};


template<typename type>
std::vector<type> unique(cv::Mat in) {
    assert(in.channels() == 1 && "This implementation is only for single-channel images");
    auto begin = in.begin<type>(), end = in.end<type>();
    auto last = std::unique(begin, end);    // remove adjacent duplicates to reduce size
    std::sort(begin, last);                 // sort remaining elements
    last = std::unique(begin, last);        // remove duplicates
    return std::vector<type>(begin, last);
}

#endif