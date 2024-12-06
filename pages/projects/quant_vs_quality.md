---
layout: page
title: Annotation Quantity vs Quality
---

As part of my researh during my PhD, understanding the effects and trade-off in dataset curation was of particular interest. This in turn resulted in a <a href="https://ieeexplore.ieee.org/abstract/document/10689528" target="_blank" style="color:blue; text-decoration: underline;">journal publication</a>, in which we investigate this. 

This is a quick & disgestible overview of the paper. 

Firstly, we introduce 3 types of annotation uncertainty; incorrect class labels, missing annotations and localisation uncertainty into the [COCO](https://cocodataset.org/#home){:target="_blank"} dataset. As shown below.

  <p style="text-align: center;">
     <img src="/assets/img/annot_exp.png" alt="Annotation Examples" style="max-width: 100%; height: auto;" />
   </p> 

Following this, we trained 3 models YOLACT (1-stage CNN), Mask-RCNN (2-stage CNN) and Mask2Former (transformer) on a variety of sample sizes for varying levels of annotation uncertainties. 


From this work we have the following key takeaways:
- All three annotation uncertainties negatively affect mAP performance, with incorrect class labels degrading the mAP performance the most, followed by missing annotations and lastly localization uncertainty.
- The results show that perfectly labeled data outperforms degraded annotations for a fixed sample size, however there is utility in adding additional data of lesser annotation quality.
- The extent of the benefits of the additional data is directly related to how degraded the annotations’ are.

