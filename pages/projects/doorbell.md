---
layout: page
title: Doorbell Analytics
---

I worked on a personal side project to incorporate analytics into our smart doorbell. While many smart doorbell providers offer this feature, it often comes with a paid subscription.

The doorbell automatically wakes when it detects motion and records for as long as motion is present or for a set period. Using these video recordings, I applied an off-the-shelf [DeepSORT](https://arxiv.org/abs/1703.07402) model to provide smart analytics for classes of interest, specifically Cars and Persons.

Here’s a step-by-step overview of the process:

1. **Date and Time Extraction**
   - The date and time are displayed in the top left corner of the video.
   - I cropped this section of the image, as it remains fixed throughout each frame.
   - After cropping, I applied a threshold and edge detection to enhance the edges for OCR.
   - The enhanced crop of the image is then processed with [Tesseract](https://github.com/tesseract-ocr/tessdoc) for Optical Character Recognition (OCR).
  
   <p style="text-align: center;">
     <img src="/assets/img/time_eg.jpg" alt="Time & Date Example" style="max-width: 100%; height: auto;" />
   </p>  
   
   <p style="text-align: center;">
     <img src="/assets/img/threshold.jpg" alt="Thresholding" style="max-width: 100%; height: auto;" />
   </p>  
   
   <p style="text-align: center;">
     <img src="/assets/img/python_func.jpg" alt="Python Function" style="max-width: 100%; height: auto;" />
   </p> 

<br>

2. **Object Detection**
  - With the date and time extracted, I used the DeepSORT model to identify & track objects of interest in each frame of the video.
  - The results are stored and exported into an Excel file.

<center>
<!-- Video Container (Responsive) -->
<div class="responsive-video">
  <iframe src="/assets/img/Doorbell_example.mp4" type="video/mp4" 
          title="YouTube video player" 
          frameborder="0" 
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
          referrerpolicy="strict-origin-when-cross-origin" 
          allowfullscreen></iframe>
</div>
</center>

<br>

3. **Data Visualization**
  - The Excel file is then imported into Tableau to provide summary statistics & graphs.

   <p style="text-align: center;">
     <img src="/assets/img/date_stats.jpg" alt="Date Statistics" style="max-width: 100%; height: auto;" />
   </p> 

   <p style="text-align: center;">
     <img src="/assets/img/day_stats.jpg" alt="Day Statistics" style="max-width: 100%; height: auto;" />
   </p> 


This approach allows for the integration of smart analytics into our doorbell without the need for a subscription service! 



