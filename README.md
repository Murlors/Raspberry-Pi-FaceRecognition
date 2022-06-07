# Raspberry-Pi-FaceRecognition
>WZU Raspberry Pi Machine Vision Programming Fundamentals and Practice Course Code  
>温州大学树莓派机器视觉编程基础与实践课程代码

定义了一个`MyApp`类，里面包含了**Flask**的一些配置，
- `face_upload`：上传人脸至数据库
- `get_image`：实时获取人脸位置并返回图片流
- `get_result`：人脸识别，并导出结果，识别的结果会存档至本地的`result`文件夹中
