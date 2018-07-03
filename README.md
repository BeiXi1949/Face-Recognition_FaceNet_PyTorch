Face-Recognition_FaceNet_PyTorch
===========================
This integrated pytorch based system is a tutorial system for those who are interseted in Computer Vision especially in face recognition. Face recognition method is using FaceNet.

Some parts of this system are copy from other Github. The sites are in the references below, appreciate their contribution.

FaceNet Models are from [Openface](https://cmusatyalab.github.io/openface/ "悬停显示")

![image](https://github.com/BeiXi1949/Face-Recognition_FaceNet_PyTorch/blob/master/test.jpeg)
****
# Environment  

On my iMac, the version of necessary environments are:

|Environment|Version|
|---|---
|Python|>=3.4
|PyTorch|0.3.0
|Torchvision|0.2.0
|Opencv|3.1.0
|Dlib|19.9.0
****
# Guide
```
Register.py is for people registion
```
___Warning:___ Please remember to press 'ESC' to log out after a 10s record video has been recorded. 

```
Recognition.py is for people recogntion
```
___Warning:___ If your device supports CPU only. Please remember to modify model loding in Line 53

|Environment|Method|
|---|---
|CPU|model.load_state_dict(torch.load('path',map_location=lambda storage, loc: storage))
|GPU|model.load_state_dict(torch.load('path'))

___Warning:___ Remember do not delete the folder './User/people_ori/Unknown'
****
# References  

|Name|Github|
|----|-----|
|Convert Torch model to PyTorch Model|[OpenFacePytorch](https://github.com/thnkim/OpenFacePytorch "悬停显示")
****
|Author|Leo（北习）|
|---|---
|E-mail|zouzijie1994@gmail.com
