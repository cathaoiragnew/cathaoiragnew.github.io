---
layout: page
title: HF hosted model 
---

Welcome to my computer vision demo! Below, you can upload an image and see the predictions overlaid by the DETR model.


## Upload Image

<script src="https://cdn.jsdelivr.net/npm/onnxjs/dist/onnx.min.js"></script>

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
    const session = await onnx.InferenceSession.create(modelURL);
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
    return new onnx.Tensor(tensor, 'float32');
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
    // For simplicity, let's assume this function returns a base64-encoded image string
    const segmentationData = results[0].data; // Get the segmentation mask

    // Convert the segmentation data into an image format (base64)
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const width = 800; // Expected width
    const height = 1066; // Expected height
    canvas.width = width;
    canvas.height = height;

    const imageData = ctx.createImageData(width, height);
    for (let i = 0; i < segmentationData.length; i++) {
      const value = Math.min(255, segmentationData[i] * 255);
      imageData.data[i * 4] = value;     // Red channel
      imageData.data[i * 4 + 1] = value; // Green channel
      imageData.data[i * 4 + 2] = value; // Blue channel
      imageData.data[i * 4 + 3] = 255;   // Alpha channel (opaque)
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL(); // Return the image in base64 format
  }
</script>

