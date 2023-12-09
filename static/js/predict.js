document.getElementById('diseaseDropdown').addEventListener('change', function() {
    const selectedDisease = this.value;

    // Use fetch API to send the selected disease to the Flask route for prediction
    fetch(`/predict?disease=${selectedDisease}`)
        .then(response => response.json())
        .then(data => {
            // Handle the prediction result returned from Flask
            console.log(data.prediction_result);
            // You can update the page or show the result in an element as needed
        })
        .catch(error => {
            console.error('Error:', error);
        });
});
