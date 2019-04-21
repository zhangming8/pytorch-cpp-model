#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/script.h>

#include <iostream>
#include <memory>
#include <string>
#include <chrono>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;
using namespace chrono;

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

int main(int argc, const char* argv[]) {
    string model_path;
    if (argc == 2){
        model_path = argv[1];
    }
    else {
        model_path = "/home/lishundong/Desktop/torch_project/example-app/model_cpp.pt";
    }
    cout << "using model:" << model_path << endl;
    string test_path = "/home/lishundong/Desktop/torch_project/pytorch-regress/data/";
    
    // init model
    int img_size = 224;  // resize img to 224
    vector<torch::jit::IValue> inputs;  //def an input
    shared_ptr<torch::jit::script::Module> module = torch::jit::load(model_path);  //load model
    module->to(at::kCUDA);  // put model to gpu
    assert(module != nullptr);
    cout << "[INFO] init model done...\n";

    int i = 0;
    double t_start, t_end, t_cost;
    t_start = getTickCount(); // get now time
    
    Mat src, image, float_image;
    for (const auto &p : fs::directory_iterator(test_path)){ //遍历文件夹中的所有文件
        string s = p.path();  //get one file path
        string suffix = s.substr(s.find_last_of(".")+1);  //获取文件的格式(后缀)
        if (suffix != "jpg"){
            continue;
        }
        cout << i << "-------------------------" << endl;
        cout << p.path() << '\n';

        src = imread(s);  //读图
	// 图像预处理 注意需要和python训练时的预处理一致
        resize(src, image, Size(img_size, img_size));  // resize 图像
        cvtColor(image, image, CV_BGR2RGB);  // bgr -> rgb
	image.convertTo(float_image, CV_32F, 1.0 / 255);   //归一化到[0,1]区间
        //cout << float_image.at<Vec3f>(100,100)[1] << endl;  //输出一个像素点点值
        auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(float_image.data, {1, img_size, img_size, 3});   //将cv::Mat转成tensor,大小为1,224,224,3
        img_tensor = img_tensor.permute({0, 3, 1, 2});  //调换顺序变为torch输入的格式 1,3,224,224
        //img_tensor[0][0] = img_tensor[0][0].sub_(0.485).div_(0.229);  //减去均值,除以标准差
        //img_tensor[0][1] = img_tensor[0][1].sub_(0.456).div_(0.224);
        //img_tensor[0][2] = img_tensor[0][2].sub_(0.406).div_(0.225);
        
        auto img_var = torch::autograd::make_variable(img_tensor, false);  //不需要梯度
	inputs.emplace_back(img_var.to(at::kCUDA));  // 把预处理后的图像放入gpu
        torch::Tensor result = module->forward(inputs).toTensor();  //前向传播获取结果
        inputs.pop_back();
        cout << "result:" << result << endl;

        auto pred = result.argmax(1);
        cout << "max index:" << pred << endl;

	/*std::tuple<torch::Tensor,torch::Tensor> res_sort = result.sort(-1, true);
	torch::Tensor top_scores = get<0>(res_sort)[0];
	torch::Tensor top_idxs = get<1>(res_sort)[0].toType(torch::kInt32);
        auto top_scores_a = top_scores.accessor<float,1>();
	auto top_idxs_a = top_idxs.accessor<int,1>();
        for (int j = 0; j < 3; ++j) {
            int idx = top_idxs_a[j];
            cout << "top-" << j+1 << " index: " << idx << ", score: " << top_scores_a[j] << endl;
        }*/

        i++;
        if (i > 1000){
            break;
        }
    }
    t_end = getTickCount();
    t_cost = t_end - t_start;
    //t_cost = t_cost / getTickFrequency();
    printf("time cost: %4.f ms\n", t_cost/1000000.0);
    return 0;
}

/*
void Mycircle(){
    Point p = Point(320, 190); //圆的中心点
    int r= 50; //圆的半径
    Scalar color = Scalar(0, 255, 0);
    circle(src, p, r, color);
}
*/
