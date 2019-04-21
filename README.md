# ubuntu16.04 c++调用pytorch模型且使用gpu预测

环境：ubuntu16.04, python3, torch==1.0.0, cuda9
首先源码安装opencv和libtorch（https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/1.0/cpp_export.md）


更改代码中的路径再执行
1. python3 create_model.py
2. sh make_script.sh 
3. cd build
4. ./predict-app
