#include "det_face_landmarks.h"


using namespace std;
using namespace cv;
using namespace dnn;


retinaface::retinaface()
{
    this->detector = readNet("/home/wangbo/3DDFA-V3-opencv-dnn/weights/retinaface_resnet50.onnx");   ////注意文件路径要写正确
    this->lmks_model = readNet("/home/wangbo/3DDFA-V3-opencv-dnn/weights/landmark.onnx");             ////注意文件路径要写正确
    this->generatePriors();

	this->det_outnames = this->detector.getUnconnectedOutLayersNames();
	this->lmks_outnames = this->lmks_model.getUnconnectedOutLayersNames();
}

void retinaface::generatePriors()
{
    const float image_size[2] = {(float)this->det_inputsize, (float)this->det_inputsize};
    this->prior_box.clear();
    for(int k=0;k<steps.size();k++)
    {
        const float* t_min_sizes = min_sizes[k];
        const int f[2] = {int(ceil(image_size[0]/steps[k])), int(ceil(image_size[1]/steps[k]))};
        for(int i=0;i<f[0];i++)
        {
            for(int j=0;j<f[1];j++)
            {
                for(int l=0;l<2;l++)
                {
                    float s_kx = t_min_sizes[l]/image_size[1];
                    float s_ky = t_min_sizes[l]/image_size[0];
                    float cx = (j+0.5)*steps[k]/image_size[1];
                    float cy = (i+0.5)*steps[k]/image_size[0];
                    if(this->clip)
                    {
                        cx = std::min(std::max(cx, 0.0f), 1.0f);
                        cy = std::min(std::max(cy, 0.0f), 1.0f);
                        s_kx = std::min(std::max(s_kx, 0.0f), 1.0f);
                        s_ky = std::min(std::max(s_ky, 0.0f), 1.0f);
                    }
                    this->prior_box.emplace_back(cv::Rect2f(cx, cy, s_kx, s_ky));
                }
            }
        }
    }
}

Mat retinaface::preprocess(const Mat& rgb_image, int& x_min_pad, int& y_min_pad)
{
    const int height = rgb_image.rows;
    const int width = rgb_image.cols;
    Mat temp_image = rgb_image.clone();
    if(height > width)
    {
        float scale = (float)width / height;
        int new_width = int(this->det_inputsize*scale);
        
        resize(rgb_image, temp_image, Size(new_width, this->det_inputsize));
    }
    else
    {
        float scale = (float)height / width;
        int new_height = int(this->det_inputsize*scale);

        resize(rgb_image, temp_image, Size(this->det_inputsize, new_height));
    }

    vector<cv::Mat> rgbChannels(3);
    split(temp_image, rgbChannels);
    for (int c = 0; c < 3; c++)
    {
        rgbChannels[c].convertTo(rgbChannels[c], CV_32FC1, 1.0 / (255.0* this->std[c]), (0.0 - this->mean[c]) / this->std[c]);
    }
    Mat input_img;
	merge(rgbChannels, input_img);
    const int padh = this->det_inputsize - temp_image.rows;
    const int padw = this->det_inputsize - temp_image.cols;
    x_min_pad = (int)floor(padw / 2.f);
    y_min_pad = (int)floor(padh / 2.f);
    cv::copyMakeBorder(input_img, input_img, y_min_pad, padh-y_min_pad, x_min_pad, padw-x_min_pad, cv::BORDER_CONSTANT, cv::Scalar(0.f, 0.f, 0.f));
    return input_img;
}

vector<Bbox> retinaface::predict(const Mat& rgb_image, const float conf_threshold, const float nms_threshold)
{
    const int original_height = rgb_image.rows;
    const int original_width = rgb_image.cols;
    int x_min_pad, y_min_pad;
    Mat img = this->preprocess(rgb_image, x_min_pad, y_min_pad);
    Mat blob = blobFromImage(img);
    this->detector.setInput(blob);
    std::vector<Mat> outs;
    this->detector.forward(outs, this->det_outnames);
    const float* conf = (float*)outs[0].data;
    const float* land = (float*)outs[1].data;
    const float* loc = (float*)outs[2].data;

    const int num_priors = this->prior_box.size();
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<vector<float>> landmarks;
    for(int i=0;i<num_priors;i++)
    {
        const float score = conf[i*2+1];
        if(score > conf_threshold)
        {
            int row_ind = i*4;
            float cx = this->prior_box[i].x + loc[row_ind]*this->variance[0]*this->prior_box[i].width;  ////decode
            float cy = this->prior_box[i].y + loc[row_ind+1]*this->variance[0]*this->prior_box[i].height;  ////decode
            float w = this->prior_box[i].width * exp(loc[row_ind+2]*this->variance[1]);   ////decode
            float h = this->prior_box[i].height * exp(loc[row_ind+3]*this->variance[1]);  ////decode
            float x = cx - w*0.5f;
            float y = cy - h*0.5f;
            x *= this->det_inputsize;
            y *= this->det_inputsize;
            w *= this->det_inputsize;
            h *= this->det_inputsize;
            boxes.emplace_back(cv::Rect((int)x, (int)y, (int)w, (int)h));
            scores.emplace_back(score);

            row_ind = i*10;
            vector<float> kpts(5*2);
            for(int j=0;j<5;j++)
            {
                float px = this->prior_box[i].x + land[row_ind+j*2]*this->variance[0]*this->prior_box[i].width;      ////decode_landm
                float py = this->prior_box[i].y + land[row_ind+j*2+1]*this->variance[0]*this->prior_box[i].height;   ////decode_landm
                px *= this->det_inputsize;
                py *= this->det_inputsize;
                kpts[j*2] = px;
                kpts[j*2+1] = py;
            }
            landmarks.emplace_back(kpts);
        }
    }

    vector<int> order = argsort_descend(scores);
    std::vector<cv::Rect> boxes_;
    std::vector<float> scores_;
    std::vector<vector<float>> landmarks_;
    std::transform(order.begin(), order.end(), std::back_inserter(boxes_), [&boxes](int i) { return boxes[i]; });       ////也可以for循环遍历元素复制
    std::transform(order.begin(), order.end(), std::back_inserter(scores_), [&scores](int i) { return scores[i]; });
    std::transform(order.begin(), order.end(), std::back_inserter(landmarks_), [&landmarks](int i) { return landmarks[i]; });
    boxes.swap(boxes_);
    scores.swap(scores_);
    landmarks.swap(landmarks_);

    vector<int> keep;
    NMSBoxes(boxes, scores, conf_threshold, nms_threshold, keep);
    const int num_keep = keep.size();
    if(num_keep==0)
    {
        cout<<"No face detected"<<endl;
        exit(-1);
    }

    const float resize_coeff = std::max(original_height, original_width) / (float)this->det_inputsize;
    vector<Bbox> bboxes(num_keep);
    for(int i=0;i<num_keep;i++)
    {
        const int ind = keep[i];
        int xmin = int(float(boxes[ind].x - x_min_pad) * resize_coeff);
        bboxes[i].xmin = std::min(std::max(xmin, 0), original_width-1);
        int ymin = int(float(boxes[ind].y - y_min_pad) * resize_coeff);
        bboxes[i].ymin = std::min(std::max(ymin, 0), original_height-1);
        int xmax = int(float(boxes[ind].x + boxes[ind].width - x_min_pad) * resize_coeff);
        bboxes[i].xmax = std::min(std::max(xmax, 0), original_width-1);
        int ymax = int(float(boxes[ind].y + boxes[ind].height - y_min_pad) * resize_coeff);
        bboxes[i].ymax = std::min(std::max(ymax, 0), original_height-1);
        for(int j=0;j<5;j++)
        {
            bboxes[i].kps[j*2] = int(float(landmarks[ind][j*2] - x_min_pad) * resize_coeff);
            bboxes[i].kps[j*2+1] = int(float(landmarks[ind][j*2+1] - y_min_pad) * resize_coeff);
        }
        bboxes[i].score = scores[ind];
    }
    return bboxes;
}

Mat retinaface::process_img(const Mat& img_resize)
{
	vector<cv::Mat> rgbChannels(3);
	split(img_resize, rgbChannels);
	for (int c = 0; c < 3; c++)
	{
		rgbChannels[c].convertTo(rgbChannels[c], CV_32FC1, 1.0 / 255.0, -this->lmk_mean[c] / 255.0);
	}
	Mat input_img;
	merge(rgbChannels, input_img);

	Mat blob = blobFromImage(input_img);
	this->lmks_model.setInput(blob);
	std::vector<Mat> outs;
	this->lmks_model.forward(outs, this->lmks_outnames);
	outs[0] *= this->lmks_inputsize;
	return outs[0];
}

vector<Mat> retinaface::infer(const Mat& rgb_image)
{
	vector<Bbox> bboxes = this->predict(rgb_image);
	const float height = rgb_image.rows;
	const float width = rgb_image.cols;
	vector<Mat> landmarks;
	for (int i = 0; i < bboxes.size(); i++)
	{
		float x1 = bboxes[i].xmin;
		float y1 = bboxes[i].ymin;
		float x2 = bboxes[i].xmax;
		float y2 = bboxes[i].ymax;
		float w = x2 - x1 + 1;
		float h = y2 - y1 + 1;

		float cx = (x2 + x1)*0.5;
		float cy = (y2 + y1)*0.5;
		float sz = max(h, w)*this->enlarge_ratio;

		x1 = cx - sz * 0.5;
		y1 = cy - sz * 0.5;
		float trans_x1 = x1;
		float trans_y1 = y1;
		x2 = x1 + sz;
		y2 = y1 + sz;

		float dx = max(0.f, -x1);
		float dy = max(0.f, -y1);
		x1 = max(0.f, x1);
		y1 = max(0.f, y1);

		float edx = max(0.f, x2 - width);
		float edy = max(0.f, y2 - height);
		x2 = min(width, x2);
		y2 = min(height, y2);

		Mat crop_img;
		rgb_image(Rect((int)x1, (int)y1, (int)x2 - (int)x1, (int)y2 - (int)y1)).copyTo(crop_img);
		if (dx > 0 || dy > 0 || edx > 0 || edy > 0)
		{
			cv::copyMakeBorder(crop_img, crop_img, int(dy), int(edy), int(dx), int(edx), BORDER_CONSTANT, Scalar(103.94, 116.78, 123.68));
		}
		resize(crop_img, crop_img, Size(this->lmks_inputsize, this->lmks_inputsize));

		Mat base_lmks = this->process_img(crop_img);
		float inv_scale = sz / this->lmks_inputsize;

		float x_min = 10000;
		float y_min = 10000;
		float x_max = -10000;
		float y_max = -10000;
		for (int idx = 0; idx < 106; idx++)
		{
			float x = base_lmks.ptr<float>(0)[idx * 2 + 0] * inv_scale + trans_x1;   ////ptr方式访问比at更高效
			float y = base_lmks.ptr<float>(0)[idx * 2 + 1] * inv_scale + trans_y1;
			x_min = std::min(x_min, x);
			y_min = std::min(y_min, y);
			x_max = std::max(x_max, x);
			y_max = std::max(y_max, y);
		}

		x1 = x_min;
		y1 = y_min;
		x2 = x_max;
		y2 = y_max;

		w = x2 - x1 + 1;
		h = y2 - y1 + 1;

		cx = (x2 + x1) * 0.5;
		cy = (y2 + y1) * 0.5;

		sz = max(h, w) * this->enlarge_ratio;
		
		x1 = cx - sz * 0.5f;
		y1 = cy - sz * 0.5f;
		trans_x1 = x1;
		trans_y1 = y1;
		x2 = x1 + sz;
		y2 = y1 + sz;

		dx = max(0.f, -x1);
		dy = max(0.f, -y1);
		x1 = max(0.f, x1);
		y1 = max(0.f, y1);

		edx = max(0.f, x2 - width);
		edy = max(0.f, y2 - height);
		x2 = min(width, x2);
		y2 = min(height, y2);

		rgb_image(Rect((int)x1, (int)y1, (int)x2 - (int)x1, (int)y2 - (int)y1)).copyTo(crop_img);
		if (dx > 0 || dy > 0 || edx > 0 || edy > 0)
		{
			cv::copyMakeBorder(crop_img, crop_img, int(dy), int(edy), int(dx), int(edx), BORDER_CONSTANT, Scalar(103.94, 116.78, 123.68));
		}
		resize(crop_img, crop_img, Size(this->lmks_inputsize, this->lmks_inputsize));

		base_lmks = this->process_img(crop_img);
		inv_scale = sz / this->lmks_inputsize;

		Mat affine_base_lmks = Mat::zeros(106, 2, CV_32FC1);
		for (int idx = 0; idx < 106; idx++)
		{
			affine_base_lmks.ptr<float>(idx)[0] = base_lmks.ptr<float>(0)[idx * 2 + 0] * inv_scale + trans_x1;   ////ptr方式访问比at更高效
			affine_base_lmks.ptr<float>(idx)[1] = base_lmks.ptr<float>(0)[idx * 2 + 1] * inv_scale + trans_y1;
		}
		landmarks.emplace_back(affine_base_lmks);
	}
	return landmarks;
}

Mat retinaface::detect(const Mat& srcimg, float* trans_params)
{
	const float H = srcimg.rows;
	Mat rgb_image;
	cvtColor(srcimg, rgb_image, COLOR_BGR2RGB);
	vector<Mat> results_all = this->infer(rgb_image);
	if (results_all.size() > 0)
	{
		Mat results = results_all[0];   ///已验证一致
		Mat landmarks(5, 2, CV_32FC1);
		const int inds[5] = { 74, 83, 54, 84, 90 };
		for (int i = 0; i < 5; i++)
		{
			landmarks.ptr<float>(i)[0] = results.ptr<float>(inds[i])[0];
			landmarks.ptr<float>(i)[1] = H - 1 - results.ptr<float>(inds[i])[1];
		}

		vector<Mat> outs = align_img(rgb_image, landmarks, this->lm3d_std, trans_params);
		return outs[0];
	}
	else
	{
		cout << "exit. no face detected! run original image. the original face image should be well cropped and resized to (224,224,3)." << endl;
		exit(-1);   ////不考虑输入224x224的情况了, 直接退出。因为从python程序看，trans_params=None时不会生成3D重建图的
	}
}