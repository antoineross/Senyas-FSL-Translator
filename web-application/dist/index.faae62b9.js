async function loadModelAndSummary() {
    const modelPath = 'model.json';
    console.log("pota3");
    const model = await tf.loadGraphModel(modelPath);
    console.log('Model loaded successfully!');
    model.summary(null, null, (summary)=>{
        document.getElementById('model-summary').textContent = summary;
    });
}
loadModelAndSummary();

//# sourceMappingURL=index.faae62b9.js.map
