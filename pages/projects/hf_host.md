---
layout: page
title: HF hosted model 
---

# Host Hugging Face Model on GitHub Pages

This page shows the result of using a Hugging Face model for image segmentation.

---

<div>
  <h3>Upload Image for Segmentation</h3>
  <input type="file" id="fileInput" onchange="loadAndSegmentImage()"/>
  <br />
  <h4>Input Image:</h4>
  <img id="inputImage" width="300" alt="Uploaded image"/>
  <br />
  <h4>Segmentation Result:</h4>
  <img id="segmentationResult" width="300" alt="Segmented result"/>
</div>

<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.12.1/dist/onnxruntime-web.min.js"></script>

<script>
  // URL for the ONNX model hosted on GitHub Releases or another CORS-enabled server
  const modelURL = "https://github.com/cathaoiragnew/cathaoiragnew.github.io/releases/download/Model/detr.onnx";

  // Function to handle image upload and prediction
  async function loadAndSegmentImage() {
    const inputFile = document.getElementById('fileInput').files[0];
    if (!inputFile) {
      alert('Please upload an image first.');
      return;
    }

    // Load the image
    const img = document.getElementById('inputImage');
    img.src = URL.createObjectURL(inputFile);

    // Load the model (use a suitable JS library to handle ONNX models)
    const session = await ort.InferenceSession.create(modelURL); // Correctly using `ort` from onnxruntime-web
    console.log("Model loaded successfully!");

    // Prepare image for inference (you can use a library to convert the image to tensor)
    const imageTensor = await prepareImageForInference(inputFile);

    // Run the model to get predictions
    const results = await session.run([imageTensor]);

    // Post-process results to show segmentation masks
    const segmentedImage = processSegmentationResults(results);

    // Show the segmented image
    const segmentationImageElement = document.getElementById('segmentationResult');
    segmentationImageElement.src = segmentedImage;
  }

  // Helper function to convert image to tensor
  async function prepareImageForInference(imageFile) {
    // This function converts the image file to a tensor that is suitable for the ONNX model
    const img = await loadImage(imageFile);
    
    // Resize and normalize image (to the expected input size for DETR, typically 800x1066)
    const tensor = preprocessImage(img, 800, 1066);
    
    // Returning a tensor that is compatible with the model
    return new ort.Tensor(tensor, 'float32'); // Using `ort.Tensor` from onnxruntime-web
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

  // Preprocess the image (resize and normalize it to fit the model input)
  function preprocessImage(img, width, height) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Resize the image to the model's expected input size
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(img, 0, 0, width, height);

    // Get image data and normalize it (assuming the model needs this format)
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    const normalizedData = new Float32Array(3 * width * height);
    
    // Normalize the data to the range [0, 1]
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i] / 255.0;
      const g = data[i + 1] / 255.0;
      const b = data[i + 2] / 255.0;
      const idx = (i / 4) * 3;
      normalizedData[idx] = r;
      normalizedData[idx + 1] = g;
      normalizedData[idx + 2] = b;
    }

    // Return a flattened array of image data
    return normalizedData;
  }

  // Post-process segmentation results to generate an image URL
  function processSegmentationResults(results) {
    // Assuming the model output includes a segmentation mask in a format we can process
    // This is where you would process the model output and convert it into an image format
    // For simplicity, let's assume this function returns a base

