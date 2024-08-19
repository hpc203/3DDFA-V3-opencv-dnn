# ifndef FACE_ANALYSIS
# define FACE_ANALYSIS
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include "utils.h"

class retinaface
{
public:
	retinaface();
	cv::Mat detect(const cv::Mat& srcimg, float* trans_params);
	std::vector<Bbox> predict(const cv::Mat& rgb_image, const float conf_threshold=0.7f, const float nms_threshold=0.4f);
private:
	const int det_inputsize = 512;
	const int lmks_inputsize = 224;

    const float variance[2] = {0.1, 0.2};
    const float mean[3] = {0.485, 0.456, 0.406};
    const float std[3] = {0.229, 0.224, 0.225};
    const float enlarge_ratio = 1.35;
    const float lmk_mean[3] = {103.94, 116.78, 123.68};
    const float lm3d_std[5 * 3] = { -0.31148657,  0.29036078,  0.13377953,  0.30979887,  0.28972036,
								0.13179526,  0.0032535 , -0.04617932,  0.55244243, -0.25216928,
								-0.38133916,  0.22405732,  0.2484662 , -0.38128236,  0.22235769 };

	const float min_sizes[3][2] = {{16.0, 32.0}, {64.0, 128.0}, {256.0, 512.0}};
    const std::vector<float> steps = {8.0, 16.0, 32.0};
    const bool clip = false;
    void generatePriors();
    std::vector<cv::Rect2f> prior_box;
    cv::Mat preprocess(const cv::Mat& rgb_image, int& x_min_pad, int& y_min_pad);
	std::vector<cv::Mat> infer(const cv::Mat& rgb_image);
	cv::Mat process_img(const cv::Mat& img_resize);

	std::vector<std::string> det_outnames;
	std::vector<std::string> lmks_outnames;
    cv::dnn::Net detector;
    cv::dnn::Net lmks_model;
};


#endif