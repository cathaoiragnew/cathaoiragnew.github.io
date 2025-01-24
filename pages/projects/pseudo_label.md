---
layout: page
title: Pseudo label frames from Video
---

[Github Link](https://github.com/cathaoiragnew/Pseudo_Label_Video_Frames)

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


Data annotation for object detection is a highly time-consuming and tedious process, as it requires precise labeling of bounding boxes around objects in each image. This meticulous task demands significant human effort, especially for large datasets, to ensure accuracy and consistency. By leveraging pseudo-labeling, where a pre-trained model generates initial annotations, this workload can be significantly reduced. These pseudo labels provide a strong starting point for annotators, allowing them to focus on refining and verifying the predictions rather than starting from scratch. This approach not only speeds up the annotation process but also improves overall efficiency and scalability in preparing high-quality datasets for object detection. For this project, I develop a pipeline for reading in frames from a video and creating pseudo labels using pretrained models on Hugging face. An overview of the process is given below.

1. Extract frames and perform quality checks on the images (extract_frames.py):
   
   - Converts the video into image frames
   - Remove any bad frames (white noise/blur)
   - Resizes the images to given dimensions
   - Save the images in jpg format with an appropriate file name
  
3. (Near)/Duplicate Detection using embeddings (duplicate_detection.py):

It is important to remove duplicates and near duplicates from the dataset as they can lead to data leakage and skew metrics of interest. Furthermore, we want a diverse dataset to help the models learn and generalize better.

   - CLIP is used to extract feature embeddings of each image
   - Using the embeddings of each image, cosine similarity is calcualted between feature embeddings
   - Using a threshold, we can filter out (near) duplicates

3.  Pseudo-Label images with HuggingFace Object Detection models (pseudo_label.py):
   
   - An Object Detection model pretrained on the COCO dataset is used to pseudo label each image
   - Each predictions is saved in the COCO json format
   - We can then filter out classes that are not considered of interest.
   - We can also filter out bounding boxes by its respective size (filter out small or extremely large boxes that may be false postives).  


Each of these steps are combined into a single python script that runs the process (create_pseudo_data.py). Below are some pseudo labelled frames from our example video.

   <p style="text-align: center;">
     <img src="/assets/img/example_1.jpg" alt="Pseudo Label Example" style="max-width: 100%; height: auto;" />
   </p>
   
   <p style="text-align: center;">
     <img src="/assets/img/example_2.jpg" alt="Pseudo Label Example" style="max-width: 100%; height: auto;" />
   </p> 

    <p style="text-align: center;">
     <img src="/assets/img/example_3.jpg" alt="Pseudo Label Example" style="max-width: 100%; height: auto;" />
   </p> 
