# ifndef FACE_RECONSTRUCTION
# define FACE_RECONSTRUCTION
#include <opencv2/dnn.hpp>
#include "utils.h"
#include "render.h"


class face_model
{
public:
	face_model(std::map<std::string, bool> args);
    myDict forward(const cv::Mat& im);
private:
	const int inputsize = 224;
    std::map<std::string, bool> args;

    //float u[107127*1];
    cv::Mat u;
    //float id[107127*80];
    cv::Mat id;
    //float exp[107127*64];
    cv::Mat exp_;
    //float u_alb[107127*1];
    cv::Mat u_alb;
    //float alb[107127*80];
    cv::Mat alb;
    int point_buf[35709*8];   ////int对应python里的np.int32, int64对应的才是python里的np.int64
    const std::vector<int> point_buf_shape = {35709, 8};
    std::vector<int> tri;
    const std::vector<int> tri_shape = {70789, 3};
    std::vector<float> uv_coords;     ///形状[35709*2];
    //float uv_coords_numpy[35709*3];
    cv::Mat uv_coords_numpy;
    // float uv_coords_torch[35709*3];
    cv::Mat uv_coords_torch;

    int ldm68[68];
    int ldm106[106];
    int ldm134[134];

    const std::vector<int> annotation_shapes = {440, 440, 380, 380, 1282, 598, 370, 31819};
    std::vector<std::vector<int>> annotation;

    const std::vector<std::vector<int>> annotation_tri_shapes = {{791, 3}, {787, 3}, {639, 3}, {639, 3}, {2405, 3}, {1074, 3}, {639, 3}, {62356, 3}};
    std::vector<std::vector<int>> annotation_tri;

    const std::vector<int> parallel_shapes = {195, 206, 130, 116, 116, 129, 154, 153, 267, 135, 85, 54, 32, 19, 12, 8, 1, 9, 15, 19, 25, 56, 81, 125, 257, 148, 155, 125, 116, 117, 126, 210, 187};
    std::vector<std::vector<int>> parallel;
    std::vector<int> v_parallel;

    //const float persc_proj[3*3] = {1015.f, 0.f, 0.f, 0.f, 1015.f, 0.f, 112.f, 112.f, 1.f};
    const cv::Mat persc_proj = (cv::Mat_<float>(3, 3) << 1015.f, 0.f, 0.f, 0.f, 1015.f, 0.f, 112.f, 112.f, 1.f);
    const float camera_distance = 10.f;
    //const float init_lit[1*1*9] = {0.8f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const cv::Mat init_lit = (cv::Mat_<float>(1, 9) << 0.8f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
    const float SH_a[3] = {PI, 2 * PI / sqrt(3.f), 2 * PI / sqrt(8.f)};
    const float SH_c[3] = {1.f/sqrt(4 * PI), sqrt(3.f) / sqrt(4.f * PI), 3 * sqrt(5.f) / sqrt(12.f * PI)};
    const cv::Mat lights = (cv::Mat_<float>(5, 6) << -1.f ,  1.f ,  1.f ,  1.7f,  1.7f,  1.7f,  1.f ,  1.f ,  1.f ,  1.7f,  1.7f,
                            1.7f, -1.f , -1.f ,  1.f ,  1.7f,  1.7f,  1.7f,  1.f , -1.f ,  1.f ,  1.7f,
                            1.7f,  1.7f,  0.f ,  0.f ,  1.f ,  1.7f,  1.7f,  1.7f);
    cv::Mat light_direction;
    std::vector<float> light_direction_norm;
    cv::Mat light_intensities;

    std::vector<std::string> outnames;
    cv::dnn::Net net_recon;
    
    std::shared_ptr<MeshRenderer_UV_cpu> uv_renderer{nullptr};
    std::shared_ptr<MeshRenderer_cpu> renderer{nullptr};

    cv::Mat compute_rotation(const cv::Mat& angles);
    cv::Mat transform(const cv::Mat& face_shape, const cv::Mat& rot, const cv::Mat& trans);
    cv::Mat to_camera(cv::Mat& face_shape);
    cv::Mat to_image(const cv::Mat& face_shape);
    cv::Mat compute_shape(const cv::Mat& alpha_id, const cv::Mat& alpha_exp);
    cv::Mat compute_albedo(const cv::Mat& alpha_alb, const bool normalize=true);
    cv::Mat compute_norm(const cv::Mat& face_shape);
    cv::Mat compute_texture(const cv::Mat& face_albedo, const cv::Mat& face_norm, const cv::Mat& alpha_sh);
    cv::Mat compute_gray_shading_with_directionlight(const cv::Mat& face_texture, const cv::Mat& normals);
    cv::Mat get_landmarks_106_2d(const cv::Mat& v2d, const cv::Mat& face_shape, const cv::Mat& angle, const cv::Mat& trans);
    std::vector<cv::Mat> segmentation(const cv::Mat& v3d);
    std::vector<cv::Mat> segmentation_visible(const cv::Mat& v3d, const std::vector<int>& visible_idx);
};


cv::Mat bilinear_interpolate(const cv::Mat& img, const cv::Mat& x, const cv::Mat& y);
cv::Mat median_filtered_color_pca(const cv::Mat& median_filtered_w, const cv::Mat& uv_color_img, const cv::Mat& uv_color_pca);
cv::Mat get_colors_from_uv(const cv::Mat& res_colors, const cv::Mat& x, const cv::Mat& y);


#endif