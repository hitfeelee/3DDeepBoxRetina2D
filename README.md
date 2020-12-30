# 3DDeepBoxRetina2D

## Summary
This project has been performed 2D object detection by Retinanet << Focal Loss for Dense Object Detection >> and 3D objects detection by 3DDeepbox << 3D Bounding Box Estimation Using Deep Learning and Geometry >>. 
Above both of sub-projects base on the lightweight backbone MobileNet-v2. of course, Replacing it with other base net conveniently, such as MobileNet-v3 and ShuffleNet-v2 etc.

![DEMO SHOW]("https://github.com/hitfee01/3DDeepBoxRetina2D/blob/master/images/detect_result.png")

if image fail to shows above, you can see it in images.  

## Quick Start
### datasets
Applying kitt dataset.
Please place it as following:

    root
    |
    ---datasets
       |
       ---data
          |
          ---kitti
              ---testing
              |
              ---training
                 ---ImageSets
                     |
                     ---train.txt // list of training image.
                     |
                     ---test.txt // list of testing image.
                 |
                 ---calib
                    |
                    ---calib_cam_to_cam.txt // camera calibration file for kitti
                 |
                 ---image_2
                 |
                 ---label_2
          

### training
    python3 TrainRetinaNet.py --model-name MOBI-V2-RETINA-FPN
    python3 TrainDeepBox3D.py --model-name MOBI-V2-DEEPBOX3D
 
### evalution
   Only performed F1score for Retinanet.
   
    python3 TrainRetinaNet.py --model-name MOBI-V2-RETINA-FPN
    
### demo
2D detection demo.

    python3 ./demos/demo_2d.py --model-name MOBI-V2-RETINA-FPN

3D detection demo without 2D detection.

    python3 ./demos/demo_3d.py --model-name MOBI-V2-DEEPBOX3D
    
3D detection demo with 2D detection.

    python3 ./demos/demo3d_with_retina.py --model-name-2D MOBI-V2-RETINA-FPN --model-name-3D MOBI-V2-DEEPBOX3D

3D detection demo with Retina3D detection.

    python3 ./demos/demo_retina3d.py --model-name MOBI-V2-RETINA3D-FPN
    
## Pretrained Model

We provide a set of trained models available for download in the  [Pretrained Model](https://pan.baidu.com/s/1KkVe2mzwp1yiL9HfjomLYQ).
提取码: uiyh
## License


## Refenrence
Detectron2.

3D-BoundingBox.