#include "utils.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace cv;


void POS(const Mat& xp, const float* x, float* t, float& s)
{
	const int npts = xp.size[0];
	Mat A = Mat::zeros(2 * npts, 8, CV_32FC1);
	for (int i = 0; i < npts; i++)
	{
		A.ptr<float>(i * 2)[0] = x[i*3];
		A.ptr<float>(i * 2)[1] = x[i*3+1];
		A.ptr<float>(i * 2)[2] = x[i*3+2];
		A.ptr<float>(i * 2)[3] = 1;

		A.ptr<float>(i * 2 + 1)[4] = x[i*3];
		A.ptr<float>(i * 2 + 1)[5] = x[i*3+1];
		A.ptr<float>(i * 2 + 1)[6] = x[i*3+2];
		A.ptr<float>(i * 2 + 1)[7] = 1;
	}

	vector<int> newshape = { 2 * npts, 1 };
	Mat b = xp.reshape(0, newshape);
	Mat k;
	cv::solve(A.t()*A, A.t()*b, k, DECOMP_CHOLESKY);
	const float* pk = (float*)k.data;
	float norm_R1 = sqrt(pk[0] * pk[0] + pk[1] * pk[1] + pk[2] * pk[2]);
	t[0] = pk[3];
	float norm_R2 = sqrt(pk[4] * pk[4] + pk[5] * pk[5] + pk[6] * pk[6]);
	t[1] = pk[7];
	s = (norm_R1 + norm_R2)*0.5f;
}

vector<Mat> resize_n_crop_img(const Mat& img, const Mat& lm, const float* t, const float s, const float target_size, const Mat& mask)
{
	const float h0 = img.rows;
	const float w0 = img.cols;
	const int w = int(w0 * s);
	const int h = int(h0 * s);
	const int left = int(w * 0.5f - target_size * 0.5f + (t[0] - w0 * 0.5f)*s);  ///避免使用除法
	const int right = std::min(int(left + target_size), w);    ///防止超出边界
	const int up = int(h * 0.5f - target_size * 0.5f + (h0 * 0.5f - t[1])*s);
	const int below = std::min(int(up + target_size), h);   ///防止超出边界

	Mat tmp_img;
	resize(img, tmp_img, Size(w, h), INTER_CUBIC);
	Mat img_new;
	Rect crop_box = Rect(left, up, right - left, below - up);
	tmp_img(crop_box).copyTo(img_new);
	const int dsth = int(target_size);
	const int dstw = int(target_size);
	if (img_new.rows < dsth)
	{
		copyMakeBorder(img_new, img_new, 0, dsth - img_new.rows, 0, 0, BORDER_CONSTANT, 0);
	}
	if (img_new.cols < dstw)
	{
		copyMakeBorder(img_new, img_new, 0, 0, 0, dstw - img_new.cols, BORDER_CONSTANT, 0);
	}

	Mat mask_new = cv::Mat();
	if (!mask.empty())
	{
		Mat tmp_mask;
		resize(mask, tmp_mask, Size(w, h), INTER_CUBIC);
		
		tmp_mask(crop_box).copyTo(mask_new);
		if (mask_new.rows < dsth)
		{
			copyMakeBorder(mask_new, mask_new, 0, dsth - mask_new.rows, 0, 0, BORDER_CONSTANT, 0);
		}
		if (mask_new.cols < dstw)
		{
			copyMakeBorder(mask_new, mask_new, 0, 0, 0, dstw - mask_new.cols, BORDER_CONSTANT, 0);
		}
	}
	Mat lm_new = lm.clone();
	for (int i = 0; i < lm.rows; i++)
	{
		lm_new.ptr<float>(i)[0] = (lm.ptr<float>(i)[0] - t[0] + w0 * 0.5f)*s - (w * 0.5f - target_size * 0.5f);
		lm_new.ptr<float>(i)[1] = (lm.ptr<float>(i)[1] - t[1] + h0 * 0.5f)*s - (h * 0.5f - target_size * 0.5f);
	}
	vector<Mat> out = { img_new, lm_new, mask_new };
	return out;
}

Mat back_resize_crop_img(const Mat& img, const float* trans_params, const Mat& ori_img, const int resample_method)
{
	const float w0 = trans_params[0];
	const float h0 = trans_params[1];
	const float s = trans_params[2];
	const float t[2] = {trans_params[3], trans_params[4]};
	const float target_size = 224.f;

	int w = int(w0 * s);
	int h = int(h0 * s);
	const int left = int(w * 0.5f - target_size * 0.5f + (t[0] - w0 * 0.5f)*s);  ///避免使用除法
	const int right = std::min(int(left + target_size), w);    ///防止超出边界
	const int up = int(h * 0.5f - target_size * 0.5f + (h0 * 0.5f - t[1])*s);
	const int below = std::min(int(up + target_size), h);   ///防止超出边界

	Mat old_img;
	resize(ori_img, old_img, Size(w, h), resample_method);
	Rect crop_box = Rect(left, up, right - left, below - up);
	
	img(Rect(0, 0, crop_box.width, crop_box.height)).copyTo(old_img(crop_box));
	resize(old_img, old_img, Size(int(w0), int(h0)), resample_method);
	return old_img;
}

void back_resize_ldms(Mat& ldms, const float* trans_params)
{
	const float w0 = trans_params[0];
	const float h0 = trans_params[1];
	const float s = trans_params[2];
	const float t[2] = {trans_params[3], trans_params[4]};
	const float target_size = 224.f;

	int w = int(w0 * s);
	int h = int(h0 * s);
	const int left = int(w * 0.5f - target_size * 0.5f + (t[0] - w0 * 0.5f)*s);  ///避免使用除法
	const int up = int(h * 0.5f - target_size * 0.5f + (h0 * 0.5f - t[1])*s);
	
	for(int i=0;i<ldms.rows;i++)
	{
		ldms.ptr<float>(i)[0] = (ldms.ptr<float>(i)[0] + left) / w * w0;
		ldms.ptr<float>(i)[1] = (ldms.ptr<float>(i)[1] + up) / h * h0;
	}
}

Mat plot_kpts(const Mat& image, const Mat& kpts, string color)
{
	Scalar c;
	if(color == "r")
	{
		c = Scalar(255, 0, 0);
	}
	else if(color == "g")
	{
		c = Scalar(0, 255, 0);
	}
	else
	{
		c = Scalar(0, 0, 255);
	}
	Mat out = image.clone();
	const int radius = max(int((float)min(image.size[0], image.size[1]) / 200.f), 1);
	for(int i=0;i<kpts.size[0];i++)
	{
		circle(out, Point(int(kpts.ptr<float>(i)[0]), int(kpts.ptr<float>(i)[1])), radius, c, radius*2);
	}
	return out;
}

Mat converthwc2bgr(const Mat& img)
{
    ////img的形状是[h,w,3], 数据类型是CV32FC1
    Mat out = Mat(img.size[0], img.size[1], CV_32FC3);
    for(int h=0;h<img.size[0];h++)
    {
        for(int w=0;w<img.size[1];w++)
        {
            out.at<Vec3f>(h,w)[0] = img.ptr<float>(h,w)[0]*255;
            out.at<Vec3f>(h,w)[1] = img.ptr<float>(h,w)[1]*255;
            out.at<Vec3f>(h,w)[2] = img.ptr<float>(h,w)[2]*255;
        }
    }
    out.convertTo(out, CV_8UC3);
    return out;
}

Mat convert_hwc2bgr(const Mat& img)
{
	////img的形状是[h,w,3], 数据类型是CV32FC1
	const int h = img.size[0];
	const int w = img.size[1];
	const int area = h*w;
	Mat img_chw;
	cv::transposeND(img, {2, 0, 1}, img_chw);
	Mat bmat = Mat(h, w, CV_32FC1, (float*)img_chw.data);
	Mat gmat = Mat(h, w, CV_32FC1, (float*)img_chw.data + area);
	Mat rmat = Mat(h, w, CV_32FC1, (float*)img_chw.data + 2*area);
	vector<Mat> channel_mats = {bmat, gmat, rmat};
	Mat dstimg;
	cv::merge(channel_mats, dstimg);

	// dstimg.convertTo(dstimg, CV_8UC3);        ///这两步要不要做, 看实际情况
	// cvtColor(dstimg, dstimg, COLOR_BGR2RGB);

	return dstimg;
}

vector<Mat> align_img(const Mat& img, const Mat& lm, const float* lm3D, float* trans_params, const Mat& mask, const float target_size, const float rescale_factor)
{
	const int h0 = img.rows;
	const int w0 = img.cols;
	Mat lm5p;
	if (lm.size[0] != 5)
	{
		////extract_5p
		const int lm_idx[7] = { 31 - 1, 37 - 1, 40 - 1, 43 - 1, 46 - 1, 49 - 1, 55 - 1 };
		lm5p.ptr<float>(0)[0] = lm.ptr<float>(lm_idx[0])[0];
		lm5p.ptr<float>(0)[1] = lm.ptr<float>(lm_idx[0])[1];

		lm5p.ptr<float>(1)[0] = (lm.ptr<float>(lm_idx[1])[0] + lm.ptr<float>(lm_idx[2])[0])*0.5f;
		lm5p.ptr<float>(1)[1] = (lm.ptr<float>(lm_idx[1])[1] + lm.ptr<float>(lm_idx[2])[1])*0.5f;

		lm5p.ptr<float>(2)[0] = (lm.ptr<float>(lm_idx[3])[0] + lm.ptr<float>(lm_idx[4])[0])*0.5f;
		lm5p.ptr<float>(2)[1] = (lm.ptr<float>(lm_idx[3])[1] + lm.ptr<float>(lm_idx[4])[1])*0.5f;

		lm5p.ptr<float>(3)[0] = lm.ptr<float>(lm_idx[5])[0];
		lm5p.ptr<float>(3)[1] = lm.ptr<float>(lm_idx[5])[1];

		lm5p.ptr<float>(4)[0] = lm.ptr<float>(lm_idx[6])[0];
		lm5p.ptr<float>(4)[1] = lm.ptr<float>(lm_idx[6])[1];
	}
	else
	{
		lm5p = lm;
	}

	float t[2];
	float s;
	POS(lm5p, lm3D, t, s);
	s = rescale_factor / s;

	vector<Mat> out = resize_n_crop_img(img, lm, t, s, target_size, mask);
	trans_params[0] = w0;
	trans_params[1] = h0;
	trans_params[2] = s;
	trans_params[3] = t[0];
	trans_params[4] = t[1];
	return out;
}

Mat split_mul_div(const Mat& render_shape, const Mat& render_mask, const Mat& img)
{
	vector<Mat> bgr_shape(3);
    split(render_shape, bgr_shape);
	vector<Mat> bgr_mask(3);
    split(render_mask, bgr_mask);
	vector<Mat> bgr_img(3);
    split(img, bgr_img);
	vector<Mat> channel_mats(3);
	for(int c=0;c<3;c++)
	{
		bgr_shape[c].convertTo(bgr_shape[c], CV_32FC1, 1/255.f);
		bgr_mask[c].convertTo(bgr_mask[c], CV_32FC1, 1/255.f);
		bgr_img[2-c].convertTo(bgr_img[2-c], CV_32FC1, 1/255.f);
		
		channel_mats[c] = bgr_shape[c].mul(bgr_mask[c]) + bgr_img[2-c].mul(1 - bgr_mask[c]);
		channel_mats[c] *= 255;
	}
	Mat result;
	merge(channel_mats, result);
    result.convertTo(result, CV_8UC3);
	cv::cvtColor(result, result, COLOR_BGR2RGB);
	return result;
}


visualizer::visualizer()
{
	this->colormap = {{0, 0, 0}, {0, 205, 0}, {0, 138, 0}, {139, 76, 57}, {139, 54, 38}, {154, 50, 205}, {72, 118, 255}, {22, 22, 139}, {255, 255, 0}};
}

Mat visualizer::show_seg_visble(const Mat& mask, const Mat& img)
{
	const float alpha = this->alphas[0];
	Mat dstimg = Mat::zeros(mask.rows, mask.cols, CV_32FC3);
	for(int index=1;index<9;index++)
	{
		Scalar pix(alpha * this->colormap[index][0], alpha * this->colormap[index][1], alpha * this->colormap[index][2]);
		dstimg.setTo(pix, mask==index);
	}

	dstimg.setTo(0, dstimg<0);
	dstimg.setTo(255, dstimg>255);
	dstimg.convertTo(dstimg, CV_8UC3);
	cv::cvtColor(dstimg, dstimg, COLOR_BGR2RGB);
	Mat dst2;
	cv::addWeighted(img, 0.5, dstimg, 0.5, 0, dst2);
	return dst2;
}

vector<Mat> visualizer::visualize_and_output(myDict result_dict, std::map<string, bool> args, const float* trans_params, const Mat& img, const string save_path)
{
	vector<string> items = {"render_shape","render_face"};
	vector<string> option_list = {"ldm68", "ldm106", "ldm106_2d", "ldm134", "seg", "seg_visible"};
	for(auto i : option_list)
	{
		if(result_dict.find(i) != result_dict.end())  ///也可以用count函数判断key是否存在
		{
			items.push_back(i);
		}
	}

	vector<Mat> visualize_list = {img};
	Mat render_shape = std::get<Mat>(result_dict["render_shape"]) * 255;
	render_shape.convertTo(render_shape, CV_8UC3);
	Mat render_face = std::get<Mat>(result_dict["render_face"]) * 255;
	render_face.convertTo(render_face, CV_8UC3);
	Mat tmp = std::get<Mat>(result_dict["render_mask"]);
	vector<Mat> channel_mats = {tmp, tmp, tmp};
	Mat render_mask;
	cv::merge(channel_mats, render_mask);
	render_mask *= 255;
	render_mask.convertTo(render_mask, CV_8UC3);

	if(trans_params != nullptr)
	{
		Mat zero_img = Mat::zeros(img.size(), img.type());
		render_shape = back_resize_crop_img(render_shape, trans_params, zero_img, 2);
		render_face = back_resize_crop_img(render_face, trans_params, zero_img, 2);
		render_mask = back_resize_crop_img(render_mask, trans_params, zero_img, 0);
	}
	
	render_shape = split_mul_div(render_shape, render_mask, img);
	visualize_list.push_back(render_shape);
	render_face = split_mul_div(render_face, render_mask, img);
	visualize_list.push_back(render_face);

	if(std::count(items.begin(), items.end(), "ldm68"))
	{
		Mat ldm68 = std::get<Mat>(result_dict["ldm68"]);
		ldm68.col(1) = 224 -1 - ldm68.col(1);
		if(trans_params != nullptr)
		{
			back_resize_ldms(ldm68, trans_params);   //原地操作, 不返回值
		}
		Mat img_ldm68 = plot_kpts(img, ldm68, "g");
		visualize_list.push_back(img_ldm68);
	}

	if(std::count(items.begin(), items.end(), "ldm106"))
	{
		Mat ldm106 = std::get<Mat>(result_dict["ldm106"]);
		ldm106.col(1) = 224 -1 - ldm106.col(1);
		if(trans_params != nullptr)
		{
			back_resize_ldms(ldm106, trans_params);   //原地操作, 不返回值
		}
		Mat img_ldm106 = plot_kpts(img, ldm106, "g");
		visualize_list.push_back(img_ldm106);
	}

	if(std::count(items.begin(), items.end(), "ldm106_2d"))
	{
		Mat ldm106_2d = std::get<Mat>(result_dict["ldm106_2d"]);
		ldm106_2d.col(1) = 224 -1 - ldm106_2d.col(1);
		if(trans_params != nullptr)
		{
			back_resize_ldms(ldm106_2d, trans_params);   //原地操作, 不返回值
		}
		Mat img_ldm106_2d = plot_kpts(img, ldm106_2d, "g");
		visualize_list.push_back(img_ldm106_2d);
	}

	if(std::count(items.begin(), items.end(), "ldm134"))
	{
		Mat ldm134 = std::get<Mat>(result_dict["ldm134"]);
		ldm134.col(1) = 224 -1 - ldm134.col(1);
		if(trans_params != nullptr)
		{
			back_resize_ldms(ldm134, trans_params);   //原地操作, 不返回值
		}
		Mat img_ldm134 = plot_kpts(img, ldm134, "g");
		visualize_list.push_back(img_ldm134);
	}

	if(std::count(items.begin(), items.end(), "seg_visible"))
	{
		vector<Mat> seg_visible = std::get<vector<Mat>>(result_dict["seg_visible"]);
		Mat new_seg_visible_one = Mat::zeros(img.rows, img.cols, CV_8UC1);
		for(int i=0;i<8;i++)
		{
			vector<Mat> channel_mats = {seg_visible[i], seg_visible[i], seg_visible[i]};
			Mat temp;
			cv::merge(channel_mats, temp);
			temp.convertTo(temp, CV_8UC3);
			if(trans_params != nullptr)
			{
				Mat zero_img = Mat::zeros(img.size(), img.type());
				temp = back_resize_crop_img(temp, trans_params, zero_img, 0);
				cvtColor(temp, temp, COLOR_BGR2RGB);
			}
    		cv::split(temp, channel_mats);
			new_seg_visible_one.setTo(i+1, channel_mats[0]==1);
		}

		Mat dstimg = this->show_seg_visble(new_seg_visible_one, img);
		visualize_list.push_back(dstimg);
	}

	if(std::count(items.begin(), items.end(), "seg"))
	{
		vector<Mat> seg = std::get<vector<Mat>>(result_dict["seg"]);
		for(int i=0;i<8;i++)
		{
			vector<Mat> channel_mats = {seg[i], seg[i], seg[i]};
			Mat temp;
			cv::merge(channel_mats, temp);
			temp.convertTo(temp, CV_8UC3);
			if(trans_params != nullptr)
			{
				Mat zero_img = Mat::zeros(img.size(), img.type());
				temp = back_resize_crop_img(temp, trans_params, zero_img, 0);
				cvtColor(temp, temp, COLOR_BGR2RGB);
			}
			Mat new_seg_i = temp * 255;
			new_seg_i.convertTo(new_seg_i, CV_8UC3);
			Mat temp2 = img.clone();
			temp2.setTo(Scalar(200,200,100), new_seg_i==255);
			visualize_list.push_back(temp2);
		}
	}

	const int len_visualize = visualize_list.size();
	const int h = img.rows;
	const int w = img.cols;
	Mat img_res;
	if(len_visualize < 4)
	{
		img_res = Mat::ones(h, len_visualize * w, CV_8UC3) * 255;
	}
	else
	{
		img_res = Mat::ones((int)ceil((float)len_visualize / 4.f) * h, 4 * w, CV_8UC3) * 255;

	}
	for(int i=0;i<len_visualize;i++)
	{
		const int row = (int)floor(i / 4.f);
		const int col = i % 4;
		const int x_start = col * w;
		const int y_start = row * h;
		const int x_end = x_start + w;
		const int y_end = y_start + h;

		Rect crop_box = Rect(x_start, y_start, x_end-x_start, y_end-y_start);
		visualize_list[i].copyTo(img_res(crop_box));
	}
	imwrite(save_path, img_res);
	return visualize_list;
}