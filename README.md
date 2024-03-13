# Teaching Courseware Assessment Dataset(TCADataset)

## About the dataset：
![image](https://github.com/aaauthors/TCDADataset/blob/main/TDAM/fig/TCDA.png)


The dataset contains teaching courseware for nine subjects common in middle school. A total of 92,624 picture samples, including four types of courseware basic visual expression elements, namely, title, text, table and picture. The details are shown in Figure 1.
The annotation information of layout elements is shown in Figure 1 (a). We define four basic categories of labels to analyze the layout structure of teaching courseware more accurately. At the top are the original pictures of courseware of different subjects. In the middle is the TCA sample comment, the color of the layout element label is:Title,Text,Table,Picture.At the bottom is the annotation example of the manual annotation method. Finally, we predict the area according to the mask of each element, so as to conduct a comprehensive evaluation on whether the teaching courseware meets the requirements of illustrated with text and pictures and form the final score.
In order to show the dataset more clearly, we counted the specific number of elements in the courseware of different subjects, as shown in Figure 1 (b).We also counted the proportion of courseware pictures of each subject in the total number of samples, as shown in Figure 1 (c).
This paper gives a detailed division of training set,validation set and test set.

## About the TCAM：
### Requirement

Python >= 3.6

Pytorch >=1.10.0
 
### Dataset Preparation

If the article is accepted for publication, you can download our prepared TCA dataset demo from "Google Drive" .We used the ResNet50+FPN backbone pre-trained on the COCO.
The file structure of the TCA dataset is as follows:

├── TCA Dataset: Dataset root directory
>>>>>├── train: Folder of all training images (64,758)

>>>>>├── val: Folder of all validation images (13892)

>>>>>├── test:Folder of all test images(13891)

>>>>>└── annotations: Corresponding labeling folder

>>>>>>>>>>>├── train.json: Annotation file corresponding to training set

>>>>>>>>>>>├── val.json: Annotation file corresponding to validation set

>>>>>>>>>>>├── test.json: Annotation file corresponding to test set    
 
              
**2.Training & Evaluation**

When training, be careful to set '--data-path' to the root directory where you store your dataset.

` python train.py --data-path /data/TCA Dataset `


Please cite this article if you use the TCA dataset：*A New Dataset and Method for Assessing Teaching Courseware*


