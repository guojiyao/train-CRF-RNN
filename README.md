# Train CRF-RNN for Semantic Image Segmentation

Martin Kersner, <m.kersner@gmail.com>,

modified by Fan Yang, <stoneyang0915@gmail.com>

This repository contains Python scripts necessary for training [CRF-RNN for Semantic Image Segmentation](https://github.com/torrvision/crfasrnn) with 20 classes. 

```bash
git clone --recursive https://github.com/stoneyang/train-CRF-RNN
```

## Prerequisites 
In order to be able to train CRF-RNN you will need to build crfrnn version of caffe from the caffe directory. If not clone this repository with `--recursive`, one can clone it from [here](https://github.com/bittnt/caffe).

## Prepare dataset for training
First, you will need images with corresponding semantic labels. The easiest way is to employ [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) dataset (!2GB) which provides those image/label pairs. **Dataset consist of 21 different classes<sup>[1](#myfootnote1)</sup>, but we treat it without the background class.**

### Download PASCAL VOC dataset
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
```

After executing commands above you can find in `VOCdevkit/VOC2012/SegmentationClass` 2913 labels and in `VOCdevkit/VOC2012/JPEGImages` their corresponding original images<sup>[2](#myfootnote2)</sup>. In order to have a better access to those directories we will create symlinks to them. Therefore, from your cloned repository you should run following commands (replace $DATASETS with your actual path where you downloaded PASCAL VOC dataset).

```bash
ln -s $DATASETS/VOCdevkit/VOC2012/SegmentationClass labels
ln -s $DATASETS/VOCdevkit/VOC2012/JPEGImages images
```

### Split classes
In the next step we have to select only images that contain classes (in our case 20, you can also remain 3 as the original version configured) for which we want to train our semantic segmentation algorithm. At first we create a list of all images that can be exploited for segmentation. 

```bash
find labels/ -printf '%f\n' | sed 's/\.png//'  | tail -n +2 > list.txt
```
Ground truth segmentations in PASCAL VOC 2012 dataset are defined as RGB images. However, if you decide to use different dataset or already preprocessed segmentations, you could be working with gray-level ones whose values exactly correspondent to label indexes in documentation. Because the workflow of creating dataset for training is separated to several parts, we access some images twice. In a case that we are working with unpreprocessed ground truth segmentations, we would have to perform conversion twice. Unfortunately, this conversion is rather time consuming (~2s), therefore we suggest to run following command first. It is not mandatory though.

```bash
python convert_labels.py labels/ list.txt converted_labels/ # OPTIONAL
```

Then we decide which classes we are interested in and specify them in *filter_images.py* (on [line 18](https://github.com/stoneyang/train-CRF-RNN/blob/stoneyang/filter_images.py#L18) there is set the 20 classes). This script will create several text files (which list images containing our desired classes) named correspondingly to selected classes. Each file has the same structure as *list.txt*. In a case of experimenting with different classes it would be wise to generate those image list for all classes from dataset.

You should be aware that if an image label is composed from more than one class in which we are interested in, that image will be always assigned to a class with lower id. This behavior could potentionally cause a problem if dataset consists of many images with the same label couples. However, this doesn't count for *background* class.

```bash
python filter_images.py labels/ list.txt lists/ # in a case you DID NOT RUN convert_labels.py script
#python filter_images.py converted_labels/ list.txt lists/ # you RUN convert_labels.py script
```


### Create LMDB database
[Original CRF-RNN](https://github.com/torrvision/crfasrnn) used for training images with size 500x500 px and we will do so as well. But if, for whatever reason, one would decide for different dimensions<sup>[3](#myfootnote3)</sup> it can be changed on [line 22](https://github.com/stoneyang/train-CRF-RNN/blob/stoneyang/data2lmdb.py#L22) of *data2lmdb.py*. Currently, we expect that the larger side in no more than 500 px. Because images/labels don't always correspond to required dimensions, we padd them with zeros in order to obtain right image/label size.

On [line 23](https://github.com/stoneyang/train-CRF-RNN/blob/stoneyang/data2lmdb.py#L23) we can set labels which we want to include into dataset.

Within training we will regularly test our network's performance. Thus, besides the training data we will need a testing data. On [line 28](https://github.com/stoneyang/train-CRF-RNN/blob/stoneyang/data2lmdb.py#L28) we can set a ratio (currently 0.1 == 10 percent of data) which denotes how much percent of data from whole dataset will be included in the test data. 

Following command will create four directories with training/testing data for images/labels.

```bash
python data2lmdb.py # in a case you DID NOT RUN convert_labels.py script
#python data2lmdb.py converted_labels/ # you RUN convert_labels.py script
```

## Training
In order to be able to start a training we will need to download precomputed weights for CRF-RNN first.

```bash
wget http://goo.gl/j7PrPZ -O TVG_CRFRNN_COCO_VOC.caffemodel
python solve.py 2>&1 | tee train.log
```

### Visualization
During training we can visualize a loss using *loss_from_log.py*. Script accepts even more than one log file. That can be useful when we had to stop training and restarted it from the last state. Therefore, we end up with two or more log files.

```bash
python loss_from_log.py train.log
```
<p align="center">
<img src="http://i.imgur.com/jlfkY1p.png?1" width=500/>
</p>

## FAQ

### I don't want to train with 3 classes. What should I do?
You have to generate lists of images for more or less classes. This is described in a paragraph above called *Split classes*. Afterward, you will also have to change prototxt description of network *TVG_CRFRNN_COCO_VOC_TRAIN_3_CLASSES.prototxt* (one can still fetch it from [here](https://github.com/martinkersner/train-CRF-RNN)). Each line in this file which contains text *CHANGED* should be modified. At each of those lines is *num_ouput: 4*, denoting 3 classes and background. 

If you want to use for example 6 different classes, you should change parameter *num_ouput* at those lines to number 7. 

<hr>
<a name="myfootnote1">(1)</a> 
aeroplane, bicycle, bird, boat, bottle, bus, car , cat, chair, cow, diningtable, dog, horse, motorbike, person, potted plant, sheep, sofa, train, tv/monitor

<a name="myfootnote2">(2)</a> 
Maybe one noticed that in `JPEGImages` directory there are more than 2913 images. This is because dataset is not used only for segmentation but also for detection.

<a name="myfootnote3">(3)</a> 
The larger dimensions of input images are, the more memory for training is required.
