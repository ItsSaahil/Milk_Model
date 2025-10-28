document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = {
        temperature: document.getElementById('temperature').value,
        ph_level: document.getElementById('ph_level').value,
        light_intensity: document.getElementById('light_intensity').value
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();
        
        // Update spoilage prediction
        const spoilageResult = document.getElementById('spoilageResult');
        spoilageResult.textContent = `${data.days_until_spoilage} days`;
        
        // Color code based on days remaining
        if (data.days_until_spoilage <= 2) {
            spoilageResult.style.color = '#e74c3c'; // Red for critical
        } else if (data.days_until_spoilage <= 4) {
            spoilageResult.style.color = '#f39c12'; // Orange for warning
        } else {
            spoilageResult.style.color = '#27ae60'; // Green for good
        }

        // Update plot
        const plotData = JSON.parse(data.plot_data);
        Plotly.newPlot('predictionPlot', plotData.data, plotData.layout);

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while making the prediction. Please try again.');
    }
});