#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/script.h>

#include <iostream>
#include <memory>
#include <string>
#include <chrono>

#include <math.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>

using namespace cv;
using namespace std;

int main(int argc, const char* argv[]) {
  int img_size = 224;
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }
  
  Mat img = imread("/home/lishundong/Desktop/torch_project/example-app/test.jpg");
  Mat resize_img;
  double t_start, t_end, t_cost;
  if (img.empty())
    {
        cout << "can not load image" << endl;
        return -1;
    }
    else
    {
        cout << "read img ..." << img.size() << endl;
    }
  resize(img, resize_img, Size(img_size, img_size));
  cout << "resized img:" << img.size() << endl;

  
  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);

  assert(module != nullptr);
  // mode to gpu
   module->to(at::kCUDA);
  std::cout << "ok\n";
  t_start = getTickCount();
  for (int i=0; i<1000; i++){
  cout << "i:" << i << endl;
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, img_size, img_size}).to(at::kCUDA)); // input img, and to gpu
  //std::cout << inputs;
  // Execute the model and turn its output into a tensor.
  at::Tensor output = module->forward(inputs).toTensor();

  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  }
  std::cout << "done ...";
  t_end = getTickCount();
  t_cost = t_end - t_start;
  printf("time cost: %2.fms\n", t_cost);
  t_cost = t_cost / getTickFrequency();
  printf("time cost: %2.fs\n", t_cost);

  return 0;
}
