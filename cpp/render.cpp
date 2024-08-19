#include "render.h"
#include "mesh_core.h"
#include "utils.h"


using namespace std;
using namespace cv;


MeshRenderer_cpu::MeshRenderer_cpu(const float rasterize_fov, const float znear, const float zfar, const int rasterize_size)
{
    const float x = tanf((rasterize_fov * 0.5) * PI / 180.f) * znear;
    Mat ndc_projection = (Mat_<float>(4, 4) << znear / x, 0.f, 0.f, 0.f, 0.f, -znear / x, 0.f, 0.f, 0.f, 0.f, -(zfar + znear) / (zfar - znear), -2.f * zfar * znear / (zfar - znear), 0.f, 0.f, -1.f, 0.f);
    this-> ndc_proj = ndc_projection * Mat::diag((Mat_<float>(4, 1) <<1.f, -1.f, -1.f, 1.f));
    this->rasterize_size = rasterize_size;
}

/*
    * @param vertex_ 3D vertices of the mesh, float型, shape=(N, 3) ,忽律batchsize
    * @param tri_ Triangles of the mesh, int型, shape=(M, 3)
    * @param feat_ Features of the mesh, float型, shape=(N, 3)
    * @param visible_vertice Whether to return the visible vertices
*/
std::tuple<Mat, Mat, Mat, vector<int>> MeshRenderer_cpu::forward(const Mat& vertex_, const int* tri, const vector<int> tri_shape, const Mat& feat_, const bool visible_vertice)
{
    Mat vertex;
    if(vertex_.size[1]==3)
    {
        copyMakeBorder(vertex_, vertex, 0, 0, 0, 1, BORDER_CONSTANT, 1);
        vertex.col(1) *= -1.f; ////避免使用for循环
    }
    else
    {
        vertex = vertex_.clone();
    }

    Mat vertex_ndc = vertex * this->ndc_proj.t();
    for(int i=0;i<vertex_ndc.rows;i++)
    {
        const float denominator = vertex_ndc.ptr<float>(i)[3];
        vertex_ndc.ptr<float>(i)[0] /= denominator;
        vertex_ndc.ptr<float>(i)[1] /= denominator;
        vertex_ndc.ptr<float>(i)[2] /= denominator;
        vertex_ndc.ptr<float>(i)[3] /= denominator;
    }

    const int c = 3;
    const int h = this->rasterize_size;
    const int w = this->rasterize_size;
    Mat vertices = vertex_ndc.colRange(0, 3).clone();
    const float ratiow = (float)w*0.5f;
    const float ratioh = (float)h*0.5f;
    vertices.col(0) *= ratiow;
    vertices.col(1) *= ratioh;
    vertices.col(0) += ratiow;
    vertices.col(1) += ratioh;
    vertices.col(2) *= -1.f;

    Mat colors = Mat::zeros(vertices.rows, vertices.cols, CV_32FC1);
    if(!feat_.empty())
    {
        colors = feat_.clone();
    }

    Mat depth_buffer = Mat::zeros(h, w, CV_32FC1)  - 999999.f;
    Mat triangle_buffer = Mat::zeros(h, w, CV_32SC1) - 1;
    const int sz[3] = {h,w,3};
    Mat barycentric_weight = Mat::zeros(3, sz, CV_32FC1);
    const int shape[3] = {h,w,c};
    Mat image = Mat::zeros(3, shape, CV_32FC1);
    _MeshRenderer_cpu_core((float*)image.data, (float*)vertices.data, tri, (float*)colors.data, (float*)depth_buffer.data, (int*)triangle_buffer.data, (float*)barycentric_weight.data, vertices.size[0], tri_shape[0], h, w, c);

    // const int area = h*w;
    // float* depth_buffer = new float[area];
    // std::fill(depth_buffer, depth_buffer + area, -999999.f);
    // int* triangle_buffer = new int[area];
    // std::fill(triangle_buffer, triangle_buffer + area, -1);
    // float* barycentric_weight = new float[area*3];
    // std::fill(barycentric_weight, barycentric_weight + area*3, 0.f);
    // float* image = new float[area*c];
    // std::fill(image, image + area*c, 0.f);
    // _MeshRenderer_cpu_core(image, (float*)vertices.data, tri, (float*)colors.data, depth_buffer, triangle_buffer, barycentric_weight, vertices.size[0], tri_shape[0], h, w, c);

    depth_buffer.setTo(0.f, depth_buffer == -999999.f);
    depth_buffer *= -1.f;

    vector<int> unique_visible_verts_idx;
    if(visible_vertice)
    {
        vector<int> visible_faces = unique<int>(triangle_buffer);
        int value = -1;
        auto filter = [&value](int val) {
            return value == val;
        };
        visible_faces.erase(std::remove_if(visible_faces.begin(), visible_faces.end(), filter), visible_faces.end());
        vector<int> visible_verts_idx(visible_faces.size()*tri_shape[1]);
        for(int i=0;i<visible_faces.size();i++)
        {
            int idx = visible_faces[i];
            for(int j=0;j<tri_shape[1];j++)
            {
                visible_verts_idx[i*tri_shape[1]+j] = tri[idx*tri_shape[1]+j];
            }
        }

        sort(visible_verts_idx.begin(), visible_verts_idx.end());
        auto pos = unique(visible_verts_idx.begin(), visible_verts_idx.end());
        visible_verts_idx.erase(pos, visible_verts_idx.end());
        unique_visible_verts_idx = visible_verts_idx;
    }

    Mat mask = Mat::zeros(triangle_buffer.rows, triangle_buffer.cols, CV_32FC1);
    mask.setTo(1, triangle_buffer > 0);
    //cv::transposeND(image, {2, 0, 1}, image);  ///不考虑batchsize

    std::tuple<Mat, Mat, Mat, vector<int>> result = std::make_tuple(mask, depth_buffer, image, unique_visible_verts_idx);
    return result;
}

MeshRenderer_UV_cpu::MeshRenderer_UV_cpu(const int rasterize_size)
{
    this->rasterize_size = rasterize_size;
}


std::tuple<Mat, Mat, Mat, vector<int>> MeshRenderer_UV_cpu::forward(const Mat& vertex_, const int* tri, const vector<int> tri_shape, const Mat& feat_, const bool visible_vertice)
{
    Mat vertex_ndc = vertex_.clone();
    vertex_ndc.col(1) *= -1.f;

    const int c = 3;
    const int h = this->rasterize_size;
    const int w = this->rasterize_size;
    Mat vertices = vertex_ndc.colRange(0, 3).clone();
    const float ratiow = (float)w*0.5f;
    const float ratioh = (float)h*0.5f;
    vertices.col(0) *= ratiow;
    vertices.col(1) *= ratioh;
    vertices.col(0) += ratiow;
    vertices.col(1) += ratioh;
    vertices.col(2) *= -1.f;

    Mat colors = Mat::zeros(vertices.rows, vertices.cols, CV_32FC1);
    if(!feat_.empty())
    {
        colors = feat_.clone();
    }

    Mat depth_buffer = Mat::zeros(h, w, CV_32FC1)  - 999999.f;
    Mat triangle_buffer = Mat::zeros(h, w, CV_32SC1) - 1;
    const int sz[3] = {h,w,3};
    Mat barycentric_weight = Mat::zeros(3, sz, CV_32FC1);
    const int shape[3] = {h,w,c};
    Mat image = Mat::zeros(3, shape, CV_32FC1);
    _MeshRenderer_cpu_core((float*)image.data, (float*)vertices.data, tri, (float*)colors.data, (float*)depth_buffer.data, (int*)triangle_buffer.data, (float*)barycentric_weight.data, vertices.size[0], tri_shape[0], h, w, c);

    depth_buffer.setTo(0.f, depth_buffer == -999999.f);
    depth_buffer *= -1.f;

    vector<int> unique_visible_verts_idx;
    if(visible_vertice)
    {
        vector<int> visible_faces = unique<int>(triangle_buffer);
        int value = -1;
        auto filter = [&value](int val) {
            return value == val;
        };
        visible_faces.erase(std::remove_if(visible_faces.begin(), visible_faces.end(), filter), visible_faces.end());
        vector<int> visible_verts_idx(visible_faces.size()*tri_shape[1]);
        for(int i=0;i<visible_faces.size();i++)
        {
            int idx = visible_faces[i];
            for(int j=0;j<tri_shape[1];j++)
            {
                visible_verts_idx[i*tri_shape[1]+j] = tri[idx*tri_shape[1]+j];
            }
        }

        sort(visible_verts_idx.begin(), visible_verts_idx.end());
        auto pos = unique(visible_verts_idx.begin(), visible_verts_idx.end());
        visible_verts_idx.erase(pos, visible_verts_idx.end());
        unique_visible_verts_idx = visible_verts_idx;
    }

    Mat mask = Mat::zeros(triangle_buffer.rows, triangle_buffer.cols, CV_32FC1);
    mask.setTo(1, triangle_buffer > 0);
    ///cv::transposeND(image, {2, 0, 1}, image);  ///不考虑batchsize

    std::tuple<Mat, Mat, Mat, vector<int>> result = std::make_tuple(mask, depth_buffer, image, unique_visible_verts_idx);
    return result;
}