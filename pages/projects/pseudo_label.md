---
layout: page
title: Pseudo labelling
---

<p style="text-align: center;">
  <a href="https://github.com/cathaoiragnew/Pseudo_Label_Video_Frames" target="_blank">
    <img src="/assets/img/git_icon_.png" alt="Workflow" style="max-width: 7.5%; height: auto;" />
  </a>
</p>
<p style="text-align: center;">Github Repo Link</p>

Example Video:

   <center>
   <!-- Video Container (Responsive) -->
   <div class="responsive-video">
     <iframe src="/assets/img/street_video_.mp4" type="video/mp4" 
             title="YouTube video player" 
             frameborder="0" 
             allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
             referrerpolicy="strict-origin-when-cross-origin" 
             allowfullscreen></iframe>
   </div>
   </center>

   <br>

- Data annotation for object detection is time-consuming, requiring precise bounding box labeling for each image
- This process is labor-intensive, especially for large datasets, to ensure accuracy and consistency
- Pseudo-labeling reduces this workload by using a pre-trained model to generate initial annotations
- Pseudo labels provide a strong starting point, enabling annotators to refine and verify predictions instead of starting from scratch
- This approach speeds up annotation, improves efficiency, and enhances scalability in preparing datasets

This project develops a pipeline that extracts frames from videos and generates pseudo labels using pre-trained models from Hugging Face, as seen below.

   <p style="text-align: center;">
     <img src="/assets/img/example_1.jpg" alt="Pseudo Label Example" style="max-width: 100%; height: auto;" />
   </p>

## Overview of process:

1. Extract frames and perform quality checks on the images (extract_frames.py):
    
    - Converts the video into image frames
    - Remove any bad frames (white noise/blur)
    - Resizes the images to given dimensions
    - Save the images in jpg format with an appropriate file name
  
2. (Near)/Duplicate Detection using embeddings (duplicate_detection.py):
    
    - It is important to remove duplicates and near duplicates from the dataset as they can lead to data leakage and skew metrics of interest. Furthermore, we want a diverse dataset to help the models learn and           generalize better. 
    - CLIP is used to extract feature embeddings of each image
    - Using the embeddings of each image, cosine similarity is calcualted between feature embeddings
    - Using a threshold, we can filter out (near) duplicates

3.  Pseudo-Label images with HuggingFace Object Detection models (pseudo_label.py):
    
    - An object detection model pretrained on the COCO dataset is used to pseudo label each image
    - Each prediction is saved in the COCO json format
    - We can then filter out COCO classes that are not considered of interest.
    - We can also filter out bounding boxes by its respective size (filter out small or extremely large boxes that may be false postives).  


Each of these steps are combined into a single python script that runs the entire process (create_pseudo_data.py). Below are some pseudo labelled frames from our example video.

   <p style="text-align: center;">
     <img src="/assets/img/run_script.png" alt="Pseudo Label Example" style="max-width: 100%; height: auto;" />
   </p>

   
   <p style="text-align: center;">
     <img src="/assets/img/example_2.jpg" alt="Pseudo Label Example" style="max-width: 100%; height: auto;" />
   </p> 

   <p style="text-align: center;">
     <img src="/assets/img/example_3.jpg" alt="Pseudo Label Example" style="max-width: 100%; height: auto;" />
   </p> 


## Lastly we provide summary statistics of the dataset

1. The script will provide class counts and distrubitions in addition to the objects size (w.r.t to COCO object size definition).




2. We provide heatmaps of the spatial distrubtion of each of the objects of interest. This 
