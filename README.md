# 说明文档

## 模块说明

### 采图模块

### 标定模块

### 推理模块（包含预处理、推理、后处理）

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/install/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/3rdparty/orbbec/lib_x64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/3rdparty/MNN/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/3rdparty/NCNN/lib/cmake/ncnn
```

## 扩展知识

### ```extern```
1. 用于声明全局变量或函数，表明该变量或函数的定义在其他文件中
2. 作用于可扩展到多个文件，避免重复定义

### ```inline```
1. 允许函数在多个文件中定义，编译时合并为一份，避免重定义错误
2. 适用于短小且频繁调用的函数，避免函数调用的额外开销
3. 适用于头文件中定义的函数，防止重定义错误

### ```static```
1. 局部函数，只在当前翻译单元 ```.cpp``` 中使用，对外不具有共享性

重定义，函数定义在头文件中，并且头文件被多个源文件包含;

1. 头文件中 ```inline``` 相关函数，允许变量或函数在多个文件中重复定义，如果函数签名相同，实现不同，具体调用哪个实现与编译器实现有关
2. 源文件中 ```extern``` 相关函数，表明相关函数在别的翻译单元定义
