#ifndef UTIL_HPP
#define UTIL_HPP
#include <iostream>
#include <map>
#include <variant>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <opencv2/core.hpp>

#define PI 3.14159265358979323846

typedef struct
{
    int xmin;
    int ymin;
    int xmax;
    int ymax;
    int kps[10];
    float score;
} Bbox;

typedef std::map<std::string, std::variant<cv::Mat, std::vector<float>, std::vector<int>, std::vector<cv::Mat>>> myDict;

template<typename T> std::vector<int> argsort_descend(const std::vector<T>& array)
{
    const int array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    std::iota(array_index.begin(), array_index.end(), 0);

    std::sort(array_index.begin(), array_index.end(),
        [&array](int pos1, int pos2) {return (array[pos1] > array[pos2]); });

    return array_index;
}


cv::Mat converthwc2bgr(const cv::Mat& img);
cv::Mat convert_hwc2bgr(const cv::Mat& img);
std::vector<cv::Mat> align_img(const cv::Mat& img, const cv::Mat& lm, const float* lm3D, float* trans_params, const cv::Mat& mask=cv::Mat(), const float target_size=224.f, const float rescale_factor=102.f);

class visualizer
{
public:
    visualizer();
    std::vector<cv::Mat> visualize_and_output(myDict result_dict, std::map<std::string, bool> args, const float* trans_params, const cv::Mat& img, const std::string save_path);
private:
    std::vector<std::vector<float>> colormap;
    const float alphas[9] = {0.75, 0.6875, 0.625, 0.5625, 0.5, 0.4375, 0.375, 0.3125, 0.25};
    cv::Mat show_seg_visble(const cv::Mat& new_seg_visible_one, const cv::Mat& img);
};

#endif