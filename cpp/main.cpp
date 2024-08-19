#include "det_face_landmarks.h"
#include "face_reconstruction.h"
#include <opencv2/highgui.hpp>


using namespace std;
using namespace cv;


int main()
{
	std::map<std::string, bool> args = {{"iscrop", true}, {"ldm68", true}, {"ldm106", true}, {"ldm106_2d", true}, {"ldm134", true}, {"seg", true}, {"seg_visible", true}, {"useTex", true}, {"extractTex", true}};
	retinaface facebox_detector;
	face_model recon_model(args);
	visualizer my_visualize;
    
	string imgpath = "/home/wangbo/3DDFA-V3-opencv-dnn/testimgs/3.jpg";    ////图片路径要写正确
	Mat srcimg = imread(imgpath);

    float trans_params[5];
	auto start_time_process = std::chrono::high_resolution_clock::now();
	Mat im = facebox_detector.detect(srcimg, trans_params);
	myDict results = recon_model.forward(im);
	auto end_time_process = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff_process = end_time_process - start_time_process;
	cout<<"#### one image Total waste time: "<<to_string(diff_process.count())<<" s"<<endl;

	string save_path = "result.jpg";
	vector<Mat> visualize_list = my_visualize.visualize_and_output(results, args, trans_params, srcimg, save_path);
	////最重要的3个结果图保存起来
	imwrite("render_shape.jpg", visualize_list[1]);
	imwrite("render_face.jpg", visualize_list[2]);
	imwrite("seg_face.jpg", visualize_list[7]);

	return 0;
}