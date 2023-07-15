const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');

async function loadModelAndSummary() {
  const modelPath = tfnode.io.fileSystem('model.json');
  console.log("pota3")
  const model = await tf.loadLayersModel(modelPath);
  console.log('Model loaded successfully!');

  model.summary(null, null, (summary) => {
    document.getElementById('model-summary').textContent = summary;
  });
}

loadModelAndSummary();
