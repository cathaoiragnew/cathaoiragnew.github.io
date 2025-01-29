---
layout: page
title: Mask R-CNN Model for Object Detection
---

# Object Detection Using Mask R-CNN

This page demonstrates the use of a Mask R-CNN ONNX model for object detection, directly on GitHub Pages.

<div>
  <h3>Upload Image for Detection</h3>
  <input type="file" id="fileInput" onchange="loadAndSegmentImage()"/>
  <br />
  <h4>Input Image:</h4>
  <img id="inputImage" width="300" alt="Uploaded image"/>
  <br />
  <h4>Detection Result:</h4>
  <img id="segmentationResult" width="300" alt="Detection result"/>
</div>

<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.10.0/dist/ort.min.js"></script>

<script>
  // URL for the ONNX model hosted on your GitHub Pages repo
  const modelURL = "https://cathaoiragnew.github.io/MaskRCNN-12-int8.onnx";  // Update with the correct URL for your model

  // Function to handle image upload and perform object detection
  async function loadAndSegmentImage() {
    const inputFile = document.getElementById('fileInput').files[0];
    if (!inputFile) {
      alert('Please upload an image first.');
      return;
    }

    // Load the image
    const img = document.getElementById('inputImage');
    img.src = URL.createObjectURL(inputFile);

    // Load the model
    try {
      const session = await ort.InferenceSession.create(modelURL);  // Correct method call using ort
      console.log("Model loaded successfully!");

      // Check the input names of the model
      console.log("Model input names:", session.inputNames);

      // Prepare image for inference
      const imageTensor = await prepareImageForInference(inputFile);

      // Run the model to get predictions
      const feeds = {};
      feeds[session.inputNames[0]] = imageTensor;  // Assuming the first input is the image input
      const results = await session.run(feeds);

      // Process results and show segmentation
      const segmentedImage = processSegmentationResults(results);
      
      // Show the segmented image
      const segmentationImageElement = document.getElementById('segmentationResult');
      segmentationImageElement.src = segmentedImage;
    } catch (error) {
      console.error("Error loading model:", error);
    }
  }

  // Helper function to convert image to tensor
  async function prepareImageForInference(imageFile) {
    const img = await loadImage(imageFile);
    const tensorData = preprocessImage(img, 800, 1066);
    
    // Reshape the tensor to have the correct dimensions: [1, 3, 800, 1066]
    const reshapedTensor = new ort.Tensor('float32', tensorData, [1, 3, 800, 1066]);  // 1 batch, 3 channels, 800x1066 image size
    
    return reshapedTensor;
  }

  // Load image into an HTMLImageElement
  function loadImage(file) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }

  // Preprocess the image (resize and normalize it)
  function preprocessImage(img, width, height) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(img, 0, 0, width, height);

    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
    const normalizedData = new Float32Array(3 * width * height);

    for (let i = 0; i < data.length; i += 4) {
      const r = data[i] / 255.0;
      const g = data[i + 1] / 255.0;
      const b = data[i + 2] / 255.0;
      const idx = (i / 4) * 3;
      normalizedData[idx] = r;
      normalizedData[idx + 1] = g;
      normalizedData[idx + 2] = b;
    }

    return normalizedData;
  }

  // Post-process segmentation results to generate an image URL
  function processSegmentationResults(results) {
    const segmentationData = results[0].data;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const width = 800;
    const height = 1066;
    canvas.width = width;
    canvas.height = height;

    const imageData = ctx.createImageData(width, height);
    for (let i = 0; i < segmentationData.length; i++) {
      const value = Math.min(255, segmentationData[i] * 255);
      imageData.data[i * 4] = value;     // Red channel
      imageData.data[i * 4 + 1] = value; // Green channel
      imageData.data[i * 4 + 2] = value; // Blue channel
      imageData.data[i * 4 + 3] = 255;   // Alpha channel
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL();
  }
</script>

