# MCMOT: One-shot multi-class multi-object tracking </br>
单阶段实时多类别多目标跟踪
</br>
This is an extention work of FairMOT, which extends the one-class multi-object tracking to multi-class multi-object tracking
</br>
You can refer to origin fork [FairMOT](https://github.com/ifzhang/FairMOT)
## Tracking demo of C5(car, bicycle, person, cyclist, tricycle)
![image](https://github.com/CaptainEven/MCMOT/blob/master/demo_c5.gif)
</br>
## Tracking demo of VISDrone dataset
![image](https://github.com/CaptainEven/MCMOT/blob/master/visdrone_31_track.gif)
</br>

## VisDrone dataset training
[VisDrone link](http://aiskyeye.com/)
</br>
VisDrone is a public dataset for 4 CV challenges: object detection, crowd counting, single class multi-object tracking, multi-class multi-object tracking.
* Download multi-class multi-object tracking part of Visdrone dataset
* Using gen_dataset_visdrone.py script to generate labels.
* Call the gen_dot_train_file function in gen_dataset_visdrone.py to generate the dot train file for VisDrone mcmot training task.
* Uncomment cls2id and id2cls in multitracker.py to use the correct class names and class Ids mapping.
```
from gen_dataset_visdrone import cls2id, id2cls  # visdrone
# from gen_labels_detrac_mcmot import cls2id, id2cls  # mcmot_c5
```
* Set cls ids for visdrone training n opts.py i.e.
```
1~10 object classes are what we need      
        non-interest-zone (0)
        pedestrian        (1)  --> 0       
        people            (2)  --> 1       
        bicycle           (3)  --> 2       
        car               (4)  --> 3       
        van               (5)  --> 4       
        truck             (6)  --> 5        
        tricycle          (7)  --> 6        
        awning-tricycle   (8)  --> 7        
        bus               (9)  --> 8        
        motor             (10) --> 9        
        others            (11)
        self.parser.add_argument('--reid_cls_ids',
                                 default='0,1,2,3,4,5,6,7,8,9',  # '0,1,2,3,4' or '0,1,2,3,4,5,6,7,8,9'
                                 help='')  # the object classes need to do reid
```

## Pretained model for C5 and VisDrone detection and tracking
HRNet18 backbone with bi-linear upsampling replaced with de-convolution </br>
The pre-trained model is for 5 classes(C5) detection & tracking: car, bicycle, person, cyclist, tricycle, which can be used for road traffic video surveillance and analysis. </br>
</br>
[baidu drive link](https://pan.baidu.com/s/1imrim0kt72_Ay9w-X4kqdw) extract code：ej4p
</br>
[one drive link](https://1drv.ms/u/s!AmgjV-Jny9CqkX2EneIkb6Q8x6WI?e=Lsqx4G)

### Resnet18 backbone for C5，which is much smaller than HRNet18
[ResNet18 one drive link](https://1drv.ms/u/s!AmgjV-Jny9Cqkjiwqx7UY2g2_uZU?e=tws0eQ)

### Resnet18 backbone for VisDrone mcmot
[Resnet18 one drive link](https://1drv.ms/u/s!AmgjV-Jny9CqkjkCXS9QC8mJwMmB?e=XzrH1T)
