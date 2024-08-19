#include"face_reconstruction.h"
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;
using namespace dnn;


Mat bilinear_interpolate(const Mat& img, const Mat& x, const Mat& y)
{
    ////img的形状是[1,3,h,w]
    const int num_pts = x.size[0];
    const int h = img.size[2];
    const int w = img.size[3];
    Mat out = Mat::zeros(num_pts, 3, CV_32FC1);
    for(int i=0;i<num_pts;i++)
    {
        int x0 = (int)floor(x.ptr<float>(i)[0]);
        int x1 = x0 + 1;
        int y0 = (int)floor(y.ptr<float>(i)[0]);
        int y1 = y0 + 1;

        x0 = std::max(std::min(x0, w-1), 0);
        x1 = std::max(std::min(x1, w-1), 0);
        y0 = std::max(std::min(y0, h-1), 0);
        y1 = std::max(std::min(y1, h-1), 0);

        float i_a[3] = {img.ptr<float>(0, 0, y0)[x0], img.ptr<float>(0, 1, y0)[x0], img.ptr<float>(0, 2, y0)[x0]};
        float i_b[3] = {img.ptr<float>(0, 0, y1)[x0], img.ptr<float>(0, 1, y1)[x0], img.ptr<float>(0, 2, y1)[x0]};
        float i_c[3] = {img.ptr<float>(0, 0, y0)[x1], img.ptr<float>(0, 1, y0)[x1], img.ptr<float>(0, 2, y0)[x1]};
        float i_d[3] = {img.ptr<float>(0, 0, y1)[x1], img.ptr<float>(0, 1, y1)[x1], img.ptr<float>(0, 2, y1)[x1]};

        float wa = (x1 - x.ptr<float>(i)[0]) * (y1 - y.ptr<float>(i)[0]);
        float wb = (x1 - x.ptr<float>(i)[0]) * (y.ptr<float>(i)[0] - y0);
        float wc = (x.ptr<float>(i)[0] - x0) * (y1 - y.ptr<float>(i)[0]);
        float wd = (x.ptr<float>(i)[0] - x0) * (y.ptr<float>(i)[0] - y0);

        out.ptr<float>(i)[0] = wa * i_a[0] + wb * i_b[0] + wc * i_c[0] + wd * i_d[0];
        out.ptr<float>(i)[1] = wa * i_a[1] + wb * i_b[1] + wc * i_c[1] + wd * i_d[1];
        out.ptr<float>(i)[2] = wa * i_a[2] + wb * i_b[2] + wc * i_c[2] + wd * i_d[2];
    }
    return out;
}

cv::Mat get_colors_from_uv(const cv::Mat& img, const cv::Mat& x, const cv::Mat& y)
{
    ////img的形状是[h,w,3]
    const int num_pts = x.size[0];
    const int h = img.size[0];
    const int w = img.size[1];
    Mat out = Mat::zeros(num_pts, 3, CV_32FC1);
    for(int i=0;i<num_pts;i++)
    {
        int x0 = (int)floor(x.ptr<float>(i)[0]);
        int x1 = x0 + 1;
        int y0 = (int)floor(y.ptr<float>(i)[0]);
        int y1 = y0 + 1;

        x0 = std::max(std::min(x0, w-1), 0);
        x1 = std::max(std::min(x1, w-1), 0);
        y0 = std::max(std::min(y0, h-1), 0);
        y1 = std::max(std::min(y1, h-1), 0);

        float i_a[3] = {img.ptr<float>(y0, x0)[0], img.ptr<float>(y0, x0)[1], img.ptr<float>(y0, x0)[2]};
        float i_b[3] = {img.ptr<float>(y1, x0)[0], img.ptr<float>(y1, x0)[1], img.ptr<float>(y1, x0)[2]};
        float i_c[3] = {img.ptr<float>(y0, x1)[0], img.ptr<float>(y0, x1)[1], img.ptr<float>(y0, x1)[2]};
        float i_d[3] = {img.ptr<float>(y1, x1)[0], img.ptr<float>(y1, x1)[1], img.ptr<float>(y1, x1)[2]};

        float wa = (x1 - x.ptr<float>(i)[0]) * (y1 - y.ptr<float>(i)[0]);
        float wb = (x1 - x.ptr<float>(i)[0]) * (y.ptr<float>(i)[0] - y0);
        float wc = (x.ptr<float>(i)[0] - x0) * (y1 - y.ptr<float>(i)[0]);
        float wd = (x.ptr<float>(i)[0] - x0) * (y.ptr<float>(i)[0] - y0);

        out.ptr<float>(i)[0] = wa * i_a[0] + wb * i_b[0] + wc * i_c[0] + wd * i_d[0];
        out.ptr<float>(i)[1] = wa * i_a[1] + wb * i_b[1] + wc * i_c[1] + wd * i_d[1];
        out.ptr<float>(i)[2] = wa * i_a[2] + wb * i_b[2] + wc * i_c[2] + wd * i_d[2];
    }
    return out;
}


Mat median_filtered_color_pca(const Mat& median_filtered_w, const Mat& uv_color_img, const Mat& uv_color_pca)
{
    /*median_filtered_w的形状是[1024,1024]，数据类型是CV32FC3
    uv_color_img和uv_color_pca的形状都是[1024,1024,3], 数据类型是CV32FC1*/
    vector<int> shape = {uv_color_img.size[0], uv_color_img.size[1], 3};
    Mat out = Mat(shape, CV_32FC1);
    for(int h=0;h<uv_color_img.size[0];h++)
    {
        for(int w=0;w<uv_color_img.size[1];w++)
        {
            float x = uv_color_img.ptr<float>(h, w)[0];
            x = std::max(std::min(x, 1.f), 0.f);
            float y = uv_color_pca.ptr<float>(h, w)[0];
            y = std::max(std::min(y, 1.f), 0.f);
            out.ptr<float>(h, w)[0] = (1 - median_filtered_w.at<Vec3f>(h,w)[0]) * x + median_filtered_w.at<Vec3f>(h,w)[0] * y;

            x = uv_color_img.ptr<float>(h, w)[1];
            x = std::max(std::min(x, 1.f), 0.f);
            y = uv_color_pca.ptr<float>(h, w)[1];
            y = std::max(std::min(y, 1.f), 0.f);
            out.ptr<float>(h, w)[1] = (1 - median_filtered_w.at<Vec3f>(h,w)[1]) * x + median_filtered_w.at<Vec3f>(h,w)[1] * y;

            x = uv_color_img.ptr<float>(h, w)[2];
            x = std::max(std::min(x, 1.f), 0.f);
            y = uv_color_pca.ptr<float>(h, w)[2];
            y = std::max(std::min(y, 1.f), 0.f);
            out.ptr<float>(h, w)[2] = (1 - median_filtered_w.at<Vec3f>(h,w)[2]) * x + median_filtered_w.at<Vec3f>(h,w)[2] * y;
        }
    }
    return out;
}

face_model::face_model(std::map<string, bool> args_)
{
    /////文件路径都要写正确
    string model_path = "/home/wangbo/3DDFA-V3-opencv-dnn/weights/net_recon.onnx";
    this->net_recon = readNet(model_path);
    this->outnames = this->net_recon.getUnconnectedOutLayersNames();
    this->args = args_;

    this->u = Mat(107127, 1, CV_32FC1);
	FILE* fp = fopen("/home/wangbo/3DDFA-V3-opencv-dnn/weights/u.bin", "rb");
	fread((float*)this->u.data, sizeof(float), this->u.total(), fp);//导入数据
	fclose(fp);//关闭文件

    this->id = Mat(107127, 80, CV_32FC1);
    fp = fopen("/home/wangbo/3DDFA-V3-opencv-dnn/weights/id.bin", "rb");
    fread((float*)this->id.data, sizeof(float), this->id.total(), fp);
    fclose(fp);

    this->exp_ = Mat(107127, 64, CV_32FC1);
    fp = fopen("/home/wangbo/3DDFA-V3-opencv-dnn/weights/exp.bin", "rb");
    fread((float*)this->exp_.data, sizeof(float), this->exp_.total(), fp);
    fclose(fp);

    this->u_alb = Mat(107127, 1, CV_32FC1);
    fp = fopen("/home/wangbo/3DDFA-V3-opencv-dnn/weights/u_alb.bin", "rb");
    fread((float*)this->u_alb.data, sizeof(float), this->u_alb.total(), fp);
    fclose(fp);

    this->alb = Mat(107127, 80, CV_32FC1);
    fp = fopen("/home/wangbo/3DDFA-V3-opencv-dnn/weights/alb.bin", "rb");
    fread((float*)this->alb.data, sizeof(float), this->alb.total(), fp);
    fclose(fp);

    int len = 35709*8;
    fp = fopen("/home/wangbo/3DDFA-V3-opencv-dnn/weights/point_buf.bin", "rb");
    fread(this->point_buf, sizeof(int), len, fp);
    fclose(fp);

    len = 70789*3;
    this->tri.resize(len);
    fp = fopen("/home/wangbo/3DDFA-V3-opencv-dnn/weights/tri.bin", "rb");
    fread(this->tri.data(), sizeof(int), len, fp);
    fclose(fp);

    len = 35709*2;
    this->uv_coords.resize(len);
    fp = fopen("/home/wangbo/3DDFA-V3-opencv-dnn/weights/uv_coords.bin", "rb");
    fread(this->uv_coords.data(), sizeof(float), len, fp);
    fclose(fp);
    ////process_uv
    const int uv_h = 1024;
    const int uv_w = 1024;
    this->uv_coords_numpy = Mat(35709, 3, CV_32FC1);
    this->uv_coords_torch = Mat(35709, 3, CV_32FC1);
    for(int i=0;i<35709;i++)
    {
        this->uv_coords_numpy.ptr<float>(i)[0] = this->uv_coords[i*2] * (uv_w - 1);
        this->uv_coords_numpy.ptr<float>(i)[1] = this->uv_coords[i*2+1] * (uv_h - 1);
        this->uv_coords_numpy.ptr<float>(i)[2] = 0;

        this->uv_coords_torch.ptr<float>(i)[0] = (this->uv_coords_numpy.ptr<float>(i)[0] / 1023.f - 0.5f)*2.f + 1e-6;
        this->uv_coords_torch.ptr<float>(i)[1] = (this->uv_coords_numpy.ptr<float>(i)[1] / 1023.f - 0.5f)*2.f + 1e-6;
        this->uv_coords_torch.ptr<float>(i)[2] = (this->uv_coords_numpy.ptr<float>(i)[2] / 1023.f - 0.5f)*2.f + 1e-6;

        this->uv_coords_numpy.ptr<float>(i)[1] = 1024 - this->uv_coords_numpy.ptr<float>(i)[1] - 1;
    }
    this->uv_renderer = std::make_shared<MeshRenderer_UV_cpu>(1024);

    len = 68;
    fp = fopen("/home/wangbo/3DDFA-V3-opencv-dnn/weights/ldm68.bin", "rb");
    fread(this->ldm68, sizeof(int), len, fp);
    fclose(fp);

    len = 106;
    fp = fopen("/home/wangbo/3DDFA-V3-opencv-dnn/weights/ldm106.bin", "rb");
    fread(this->ldm106, sizeof(int), len, fp);
    fclose(fp);

    len = 134;
    fp = fopen("/home/wangbo/3DDFA-V3-opencv-dnn/weights/ldm134.bin", "rb");
    fread(this->ldm134, sizeof(int), len, fp);
    fclose(fp);

    for(int i=0;i<this->annotation_shapes.size();i++)
    {
        len = this->annotation_shapes[i];
        vector<int> anno(len);
        string binpath = "/home/wangbo/3DDFA-V3-opencv-dnn/weights/annotation_"+to_string(i)+".bin";
        fp = fopen(binpath.c_str(), "rb");
        fread(anno.data(), sizeof(int), len, fp);
        fclose(fp);

        this->annotation.emplace_back(anno);
    }

    for(int i=0;i<this->annotation_tri_shapes.size();i++)
    {
        len = this->annotation_tri_shapes[i][0]*this->annotation_tri_shapes[i][1];
        vector<int> anno_tri(len);
        string binpath = "/home/wangbo/3DDFA-V3-opencv-dnn/weights/annotation_tri_"+to_string(i)+".bin";
        fp = fopen(binpath.c_str(), "rb");
        fread(anno_tri.data(), sizeof(int), len, fp);
        fclose(fp);

        this->annotation_tri.emplace_back(anno_tri);
    }

    for(int i=0;i<this->parallel_shapes.size();i++)
    {
        len = this->parallel_shapes[i];
        vector<int> para(len);
        string binpath = "/home/wangbo/3DDFA-V3-opencv-dnn/weights/parallel_"+to_string(i)+".bin";
        fp = fopen(binpath.c_str(), "rb");
        fread(para.data(), sizeof(int), len, fp);
        fclose(fp);

        this->parallel.emplace_back(para);
    }

    this->v_parallel.resize(35709, -1);
    for(int i=0;i<this->parallel.size();i++)
    {
        for(int j=0;j<this->parallel[i].size();j++)
        {
            this->v_parallel[this->parallel[i][j]] = i;
        }
    }

    const float rasterize_fov = 2 * atan(112. / 1015) * 180 / PI;
    this->renderer = std::make_shared<MeshRenderer_cpu>(rasterize_fov, 5.f, 15.f, 2*112);

    this->light_direction = this->lights.colRange(0,3);
    this->light_intensities = this->lights.colRange(3, this->lights.size[1]);
    this->light_direction_norm.resize(this->light_direction.size[0]);
    for(int i=0;i<this->light_direction.size[0];i++)
    {
        this->light_direction_norm[i] = cv::norm(this->light_direction.row(i));
    }
}

Mat face_model::compute_rotation(const Mat& angles)
{
    float x = angles.ptr<float>(0)[0];
    float y = angles.ptr<float>(0)[1];
    float z = angles.ptr<float>(0)[2];
    Mat rot_x = (Mat_<float>(3, 3) << 1, 0, 0, 0, cos(x), -sin(x), 0, sin(x), cos(x));
    Mat rot_y = (Mat_<float>(3, 3) << cos(y), 0, sin(y), 0, 1, 0, -sin(y), 0, cos(y));
    Mat rot_z = (Mat_<float>(3, 3) << cos(z), -sin(z), 0, sin(z), cos(z), 0, 0, 0, 1);
    
    Mat rot = rot_z * rot_y * rot_x;
    rot = rot.t();
    // vector<int> newshape = {1, 3, 3};   ////由于在c++的opencv里不支持3维Mat的矩阵乘法,此处不考虑batchsize维度
    // rot.reshape(0, newshape);
    return rot;
}

Mat face_model::transform(const Mat& face_shape, const Mat& rot, const Mat& trans)
{
    Mat out = face_shape * rot + cv::repeat(trans, face_shape.size[0], 1);   ///c++里没有numpy的广播机制,所以这里需要用repeat
    return out;
}

Mat face_model::to_camera(Mat& face_shape)
{
    face_shape.col(2) = this->camera_distance - face_shape.col(2);
    return face_shape;
}

Mat face_model::to_image(const Mat& face_shape)
{
    Mat face_proj = face_shape * this->persc_proj;
    face_proj.colRange(0, 2) = face_proj.colRange(0, 2) / cv::repeat(face_proj.col(2), 1, 2);
    return face_proj.colRange(0, 2);
}

Mat face_model::compute_shape(const Mat& alpha_id, const Mat& alpha_exp)
{
    Mat face_shape = alpha_id * this->id.t() + alpha_exp * this->exp_.t() + this->u.t();
    int len = face_shape.size[1] / 3;
    vector<int> newshape = {len, 3};   ////由于在c++的opencv里不支持3维Mat的矩阵乘法,此处不考虑batchsize维度
    face_shape = face_shape.reshape(0, newshape);
    return face_shape;
}

Mat face_model::compute_albedo(const Mat& alpha_alb, const bool normalize)
{
    Mat face_albedo = alpha_alb * this->alb.t() + this->u_alb.t();
    if(normalize)
    {
        face_albedo /= 255.f;
    }
    int len = face_albedo.size[1] / 3;
    vector<int> newshape = {len, 3};   ////由于在c++的opencv里不支持3维Mat的矩阵乘法,此处不考虑batchsize维度
    face_albedo = face_albedo.reshape(0, newshape);
    return face_albedo;
}

Mat face_model::compute_norm(const Mat& face_shape)
{
    Mat face_norm = Mat::zeros(this->tri_shape[0]+1, 3, CV_32FC1);
    for(int i=0;i<this->tri_shape[0];i++)
    {
        Mat v1 = face_shape.row(this->tri[i*3]);
        Mat v2 = face_shape.row(this->tri[i*3+1]);
        Mat v3 = face_shape.row(this->tri[i*3+2]);
        Mat e1 = v1 - v2;
        Mat e2 = v2 - v3;
        Mat n = e1.cross(e2);

        face_norm.row(i) = n / cv::norm(n);
    }
    
    Mat vertex_norm = Mat::zeros(this->point_buf_shape[0], 3, CV_32FC1);
    for(int i=0;i<this->point_buf_shape[0];i++)
    {
        Mat vs = Mat::zeros(this->point_buf_shape[1], 3, CV_32FC1);
        for(int j=0;j<this->point_buf_shape[1];j++)
        {
            face_norm.row(this->point_buf[i*this->point_buf_shape[1]+j]).copyTo(vs.row(j));
        }
        Mat vs_colsum;
        cv::reduce(vs, vs_colsum, 0, cv::REDUCE_SUM, CV_32FC1);   ///沿着列求和
        vertex_norm.row(i) = vs_colsum / cv::norm(vs_colsum);
    }
    return vertex_norm;
}

Mat face_model::compute_texture(const Mat& face_albedo, const Mat& face_norm, const Mat& alpha_sh)
{
    Mat alpha_sh_ = alpha_sh.reshape(0, {3, 9});   ////不考虑batchsize维度
    alpha_sh_ += cv::repeat(this->init_lit, 3, 1);
    alpha_sh_ = alpha_sh_.t();
    Mat Y = Mat::zeros(face_norm.size[0], 9, CV_32FC1);
    Y.col(0) = this->SH_a[0] * this->SH_c[0] * Mat::ones(face_norm.size[0], 1, CV_32FC1);
    Y.col(1) = -this->SH_a[1] * this->SH_c[1] * face_norm.col(1);
    Y.col(2) = this->SH_a[1] * this->SH_c[1] * face_norm.col(2);
    Y.col(3) = -this->SH_a[1] * this->SH_c[1] * face_norm.col(0);
    Y.col(4) = this->SH_a[2] * this->SH_c[2] * face_norm.col(0).mul(face_norm.col(1));  ///python里的*是对应元素相乘,@是矩阵乘法, 而c++里的*是矩阵乘法,mul是对应元素相乘
    Y.col(5) = -this->SH_a[2] * this->SH_c[2] * face_norm.col(1).mul(face_norm.col(2));
    Y.col(6) = 0.5 * this->SH_a[2] * this->SH_c[2] / sqrt(3.f) * (3 * face_norm.col(2).mul(face_norm.col(2)) - 1.f);
    Y.col(7) = -this->SH_a[2] * this->SH_c[2] * face_norm.col(0).mul(face_norm.col(2));
    Y.col(8) = 0.5 * this->SH_a[2] * this->SH_c[2] * (face_norm.col(2).mul(face_norm.col(2))  - face_norm.col(1).mul(face_norm.col(1)));
    
    Mat face_texture = Mat::zeros(face_albedo.size[0], 3, CV_32FC1);
    face_texture.col(0) = Y * alpha_sh_.col(0);
    face_texture.col(1) = Y * alpha_sh_.col(1);
    face_texture.col(2) = Y * alpha_sh_.col(2);
    face_texture = face_texture.mul(face_albedo);
    return face_texture;
}

Mat face_model::compute_gray_shading_with_directionlight(const Mat& face_texture, const Mat& normals)
{
    Mat texture = Mat(normals.size[0], 3, CV_32FC1);   ///不考虑batchsize维度
    for(int i=0;i<normals.size[0];i++)
    {
        float sum_[3] = {0.f, 0.f, 0.f};
        for(int j=0;j<this->lights.size[0];j++)
        {
            const float x = this->light_direction_norm[j];
            float y = cv::sum(normals.row(i).mul(this->light_direction.row(j) / x))[0];
            y = std::max(std::min(y, 1.f), 0.f);
            sum_[0] += (y*this->light_intensities.ptr<float>(j)[0]);
            sum_[1] += (y*this->light_intensities.ptr<float>(j)[1]);
            sum_[2] += (y*this->light_intensities.ptr<float>(j)[2]);
        }
        texture.ptr<float>(i)[0] = face_texture.ptr<float>(i)[0] * (sum_[0] / this->lights.size[0]);
        texture.ptr<float>(i)[1] = face_texture.ptr<float>(i)[1] * (sum_[1] / this->lights.size[0]);
        texture.ptr<float>(i)[2] = face_texture.ptr<float>(i)[2] * (sum_[2] / this->lights.size[0]);
    }
    return texture;
}

Mat face_model::get_landmarks_106_2d(const Mat& v2d, const Mat& face_shape, const Mat& angle, const Mat& trans)
{
    Mat temp_angle = angle.clone();
    temp_angle.ptr<float>(0)[2] = 0;
    Mat rotation_without_roll = this->compute_rotation(temp_angle);
    Mat temp = this->transform(face_shape, rotation_without_roll, trans);
    Mat v2d_without_roll = this->to_image(this->to_camera(temp));

    Mat visible_parallel = Mat(this->v_parallel.size(), 1, CV_32SC1, this->v_parallel.data());
    vector<int> ldm106_dynamic(this->ldm106, this->ldm106+106);
    for(int i=0;i<16;i++)
    {
        Mat temp = v2d_without_roll.col(0).clone();
        temp.setTo(1e5, visible_parallel!=i);
        ldm106_dynamic[i] = std::min_element(temp.begin<float>(), temp.end<float>()) - temp.begin<float>();
    }

    for(int i=17;i<33;i++)
    {
        Mat temp = v2d_without_roll.col(0).clone();
        temp.setTo(-1e5, visible_parallel!=i);
        ldm106_dynamic[i] = std::max_element(temp.begin<float>(), temp.end<float>()) - temp.begin<float>();
    }
    Mat out = Mat(106, 2, CV_32FC1);
    for(int i=0;i<106;i++)
    {
        out.ptr<float>(i)[0] = v2d.ptr<float>(ldm106_dynamic[i])[0];
        out.ptr<float>(i)[1] = v2d.ptr<float>(ldm106_dynamic[i])[1];
    }
    return out;
}

vector<Mat> face_model::segmentation(const Mat& v3d)
{
    // const int sz[3] = {224,224,8};
    // Mat seg = Mat::zeros(3, sz, CV_32FC1);
    vector<Mat> seg(8);
    for(int i=0;i<8;i++)
    {
        std::tuple<Mat, Mat, Mat, vector<int>> render_outs = this->renderer->forward(v3d, this->annotation_tri[i].data(), this->annotation_tri_shapes[i]);
        seg[i] = std::get<0>(render_outs);
    }
    return seg;
}

vector<Mat> face_model::segmentation_visible(const Mat& v3d, const vector<int>& visible_idx)
{
    vector<Mat> seg(8);
    for(int i=0;i<8;i++)
    {
        Mat temp = Mat::zeros(v3d.size[0], v3d.size[1], CV_32FC1);
        for(int j=0;j<this->annotation[i].size();j++)
        {
            temp.row(this->annotation[i][j]) = 1;
        }
        for(int j=0;j<visible_idx.size();j++)
        {
            if(visible_idx[j] == 0)
            {
                temp.row(j) = 0;
            }
        }
        std::tuple<Mat, Mat, Mat, vector<int>> render_outs = this->renderer->forward(v3d, this->tri.data(), this->tri_shape, temp);
        Mat temp_image = std::get<2>(render_outs);
        Mat temp_mask = Mat(temp_image.size[0], temp_image.size[1], CV_32FC1);
        for(int h=0;h<temp_image.size[0];h++)
        {
            for(int w=0;w<temp_image.size[1];w++)
            {
                float sum = 0.f;
                for(int c=0;c<temp_image.size[2];c++)
                {
                    sum += temp_image.ptr<float>(h,w)[c];  
                }
                temp_mask.ptr<float>(h)[w] = sum / (float)temp_image.size[2];
            }
        }
        Mat mask = Mat::zeros(temp_mask.size[0], temp_mask.size[1], CV_32FC1);
        mask.setTo(1, temp_mask >= 0.5);
        mask.copyTo(seg[i]);
    }
    return seg;
}

myDict face_model::forward(const Mat& im)
{
    Mat input_img;
    im.convertTo(input_img, CV_32FC3, 1/255.f);
    Mat blob = blobFromImage(input_img);
	this->net_recon.setInput(blob);
	std::vector<Mat> outs;
	this->net_recon.forward(outs, this->outnames);
    Mat alpha = outs[0];

    ////split_alpha
    std::map<string, Mat> alpha_dict = {{"id", alpha.colRange(0, 80)}, {"exp", alpha.colRange(80, 144)}, 
                                        {"alb", alpha.colRange(144, 224)}, {"angle", alpha.colRange(224, 227)},
                                        {"sh", alpha.colRange(227, 254)}, {"trans", alpha.colRange(254, alpha.size[1])}};
    ////compute_shape
    Mat face_shape = this->compute_shape(alpha_dict["id"], alpha_dict["exp"]);
    Mat rotation = this->compute_rotation(alpha_dict["angle"]);
    Mat v3d = this->transform(face_shape, rotation, alpha_dict["trans"]);

    v3d = this->to_camera(v3d);

    Mat v2d =this->to_image(v3d);

    Mat face_albedo = this->compute_albedo(alpha_dict["alb"]);
    Mat face_norm = this->compute_norm(face_shape);
    Mat face_norm_roted = face_norm * rotation;
    Mat face_texture = this->compute_texture(face_albedo, face_norm_roted, alpha_dict["sh"]);

    face_texture.setTo(0, face_texture < 0);
    face_texture.setTo(1, face_texture > 1);

    std::tuple<Mat, Mat, Mat, vector<int>> render_outs = this->renderer->forward(v3d, this->tri.data(), this->tri_shape, face_texture, true);
    Mat pred_image = std::get<2>(render_outs);

    vector<int> visible_idx_renderer = std::get<3>(render_outs);

    Mat gray_shading = this->compute_gray_shading_with_directionlight(Mat::ones(face_albedo.rows, face_albedo.cols, CV_32FC1)*0.78, face_norm_roted);
    std::tuple<Mat, Mat, Mat, vector<int>> render_outs2 = this->renderer->forward(v3d, this->tri.data(), this->tri_shape, gray_shading);
    Mat mask = std::get<0>(render_outs2);
    Mat pred_image_shape = std::get<2>(render_outs2);

    myDict result_dict;   ///也可以使用结构体
    result_dict = {{"v3d", v3d}, {"v2d", v2d}, 
                    {"face_texture", face_texture}, {"tri", this->tri}, 
                    {"uv_coords", this->uv_coords}, {"render_shape", convert_hwc2bgr(pred_image_shape)}, 
                    {"render_face", convert_hwc2bgr(pred_image)}, {"render_mask", mask}};

    vector<int> visible_idx(35709, 0);
    if(this->args["seg_visible"] || this->args["extractTex"])
    {
        for(int i=0;i<visible_idx_renderer.size();i++)
        {
            visible_idx[visible_idx_renderer[i]] = 1;
        }
        for(int i=0;i<face_norm_roted.size[0];i++)
        {
            if(face_norm_roted.ptr<float>(i)[2] < 0)
            {
                visible_idx[i] = 0;
            }
        }
    }

    if(this->args["ldm68"])
    {
        /////get_landmarks_68
        Mat v2d_68 = Mat(68, 2, CV_32FC1);
        for(int i=0;i<68;i++)
        {
            v2d_68.ptr<float>(i)[0] = v2d.ptr<float>(this->ldm68[i])[0];
            v2d_68.ptr<float>(i)[1] = v2d.ptr<float>(this->ldm68[i])[1];
        }
        result_dict["ldm68"] = v2d_68;
    }

    if(this->args["ldm106"])
    {
        /////get_landmarks_106
        Mat v2d_106 = Mat(106, 2, CV_32FC1);
        for(int i=0;i<106;i++)
        {
            v2d_106.ptr<float>(i)[0] = v2d.ptr<float>(this->ldm106[i])[0];
            v2d_106.ptr<float>(i)[1] = v2d.ptr<float>(this->ldm106[i])[1];
        }
        result_dict["ldm106"] = v2d_106;
    }

    if(this->args["ldm106_2d"])
    {
        Mat v2d_106_2d = this->get_landmarks_106_2d(v2d, face_shape, alpha_dict["angle"], alpha_dict["trans"]);
        result_dict["ldm106_2d"] = v2d_106_2d;
    }

    if(this->args["ldm134"])
    {
        /////get_landmarks_134
        Mat v2d_134 = Mat(134, 2, CV_32FC1);
        for(int i=0;i<134;i++)
        {
            v2d_134.ptr<float>(i)[0] = v2d.ptr<float>(this->ldm134[i])[0];
            v2d_134.ptr<float>(i)[1] = v2d.ptr<float>(this->ldm134[i])[1];
        }
        result_dict["ldm134"] = v2d_134;
    }

    if(this->args["seg"])
    {
        vector<Mat> seg = this->segmentation(v3d);
        result_dict["seg"] = seg;
    }

    if(this->args["seg_visible"])
    {
        vector<Mat> seg_visible = this->segmentation_visible(v3d, visible_idx);
        result_dict["seg_visible"] = seg_visible;
    }

    if(this->args["extractTex"])
    {
        std::tuple<Mat, Mat, Mat, vector<int>> uv_renderer_outs = this->uv_renderer->forward(this->uv_coords_torch, this->tri.data(), this->tri_shape, face_texture);
        Mat uv_color_pca = std::get<2>(uv_renderer_outs);
        Mat img_colors = bilinear_interpolate(blob, v2d.col(0), v2d.col(1));
        const vector<int> newshape = {1, img_colors.size[0], img_colors.size[1]};
        std::tuple<Mat, Mat, Mat, vector<int>> uv_renderer_outs2 = this->uv_renderer->forward(this->uv_coords_torch, this->tri.data(), this->tri_shape, img_colors.reshape(0, newshape));
        Mat uv_color_img = std::get<2>(uv_renderer_outs2);
        Mat visible_idx_mat = cv::repeat(Mat(visible_idx.size(), 1, CV_32SC1, visible_idx.data()), 1, 3);
        visible_idx_mat.convertTo(visible_idx_mat, CV_32FC1);
        std::tuple<Mat, Mat, Mat, vector<int>> uv_renderer_outs3 = this->uv_renderer->forward(this->uv_coords_torch, this->tri.data(), this->tri_shape, 1-visible_idx_mat);
        Mat uv_weight = std::get<2>(uv_renderer_outs3);
        
        Mat median_filtered_w;
        cv::medianBlur(converthwc2bgr(uv_weight), median_filtered_w, 31);
        median_filtered_w.convertTo(median_filtered_w, CV_32FC3, 1/255.f);

        Mat res_colors = median_filtered_color_pca(median_filtered_w, uv_color_img, uv_color_pca);
        Mat v_colors = get_colors_from_uv(res_colors, this->uv_coords_numpy.col(0), this->uv_coords_numpy.col(1));
        result_dict["extractTex"] = v_colors;
    }
    cout<<"forward done"<<endl;
    return result_dict;
}