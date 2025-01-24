---
layout: page
title: Pseudo label frames from Videos 
---

Data annotation for object detection is a highly time-consuming and tedious process, as it requires precise labeling of bounding boxes around objects in each image. This meticulous task demands significant human effort, especially for large datasets, to ensure accuracy and consistency. By leveraging pseudo-labeling, where a pre-trained model generates initial annotations, this workload can be significantly reduced. These pseudo labels provide a strong starting point for annotators, allowing them to focus on refining and verifying the predictions rather than starting from scratch. This approach not only speeds up the annotation process but also improves overall efficiency and scalability in preparing high-quality datasets for object detection. For this project, I develop a pipeline for reading in frames from a video and creating pseudo labels using pretrained models on Hugging face. An overview of the process is given below.

1. Extract frames and perform quality checks on the images (extract_frames.py):
   - Converts the video into image frames
   - Remove any bad frames (white noise/blur)
   - Resizes the images to given dimensions
   - Save the images in jpg format with an appropriate file name
  
2. (Near)/Duplicate Detection (duplicate_detection.py):
   It is important to remove duplicates and near duplicates from our dataset as they can lead to data leakage and skew metrics of interest. Furthermore we want a diverse dataset for our models to better generalize. 
