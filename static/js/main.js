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

        if (!response.ok) {
            // Try to extract JSON error message, otherwise text
            let errText = '';
            try { errText = (await response.json()).error || JSON.stringify(await response.json()); } catch (e) { errText = await response.text(); }
            throw new Error(errText || response.statusText || 'Server returned an error');
        }

        const data = await response.json();
        
        // Debug: log the response data
        console.log('Server response:', data);
        
        // Update spoilage prediction
        const spoilageResult = document.getElementById('spoilageResult');
        spoilageResult.textContent = `${data.days_until_spoilage} days`;
        
        // Color code based on days remaining
        if (data.days_until_spoilage <= 1) {
            spoilageResult.style.color = '#e74c3c'; // Red for critical
        } else if (data.days_until_spoilage <= 2) {
            spoilageResult.style.color = '#f39c12'; // Orange for warning
        } else if (data.days_until_spoilage <= 4) {
            spoilageResult.style.color = '#e67e22'; // Orange-red for medium
        } else {
            spoilageResult.style.color = '#27ae60'; // Green for good
        }

        // Update risk level and score
        const riskLevel = document.getElementById('riskLevel');
        const riskScore = document.getElementById('riskScore');
        const riskBar = document.getElementById('riskBar');
        
        // Safe access with fallback
        const riskLevelValue = data.risk_level || 'Unknown';
        const riskScoreValue = data.risk_score || 0;
        
        riskLevel.textContent = riskLevelValue;
        riskLevel.className = 'risk-value ' + (riskLevelValue ? riskLevelValue.toLowerCase() : 'unknown');
        riskScore.textContent = riskScoreValue;
        riskBar.style.width = (riskScoreValue * 10) + '%';

        // Update current conditions
        document.getElementById('condTemp').textContent = data.current_conditions.temperature + '°C';
        document.getElementById('condPH').textContent = data.current_conditions.ph_level;
        document.getElementById('condLight').textContent = data.current_conditions.light_intensity + ' lux';

        // Update recommendations
        const recList = document.getElementById('recommendationsList');
        recList.innerHTML = '';
        if (data.recommendations.length > 0) {
            data.recommendations.forEach(rec => {
                const li = document.createElement('li');
                li.textContent = rec;
                recList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No specific recommendations at this time.';
            recList.appendChild(li);
        }

        // Update preventions
        const prevList = document.getElementById('preventionsList');
        prevList.innerHTML = '';
        if (data.preventions.length > 0) {
            data.preventions.forEach(prev => {
                const li = document.createElement('li');
                li.textContent = prev;
                prevList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No critical prevention steps needed.';
            prevList.appendChild(li);
        }

        // Update optimal conditions
        const optimalDiv = document.getElementById('optimalConditions');
        optimalDiv.innerHTML = '';
        Object.entries(data.optimal_conditions).forEach(([key, value]) => {
            const condDiv = document.createElement('div');
            const keyFormatted = key.replace(/_/g, ' ');
            condDiv.innerHTML = `<strong>${keyFormatted}:</strong> <em>${value}</em>`;
            optimalDiv.appendChild(condDiv);
        });

        // Update plot
        const plotData = JSON.parse(data.plot_data);
        Plotly.newPlot('predictionPlot', plotData.data, plotData.layout);

        // Enable download button
        const downloadBtn = document.getElementById('downloadBtn');
        downloadBtn.style.display = 'inline-block';
        downloadBtn.dataset.days = data.days_until_spoilage;

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while making the prediction. Please try again.\n\nError: ' + error.message);
    }
});

// Download enriched CSV
document.getElementById('downloadBtn').addEventListener('click', async (e) => {
    e.preventDefault();

    const temperature = document.getElementById('temperature').value;
    const ph_level = document.getElementById('ph_level').value;
    const light_intensity = document.getElementById('light_intensity').value;
    const days_until_spoilage = e.target.dataset.days;

    const downloadData = {
        temperature: parseFloat(temperature),
        ph_level: parseFloat(ph_level),
        light_intensity: parseFloat(light_intensity),
        days_until_spoilage: parseInt(days_until_spoilage)
    };

    try {
        const response = await fetch('/download-enriched', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(downloadData)
        });

        if (!response.ok) {
            throw new Error('Failed to download file');
        }

        // Get the filename from the response headers
        const contentDisposition = response.headers.get('content-disposition');
        let filename = 'milk_analysis.csv';
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/download_name=(.+)/);
            if (filenameMatch) filename = filenameMatch[1].replace(/"/g, '');
        }

        // Create a blob and download
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(link);

        // Show success message
        const downloadStatus = document.getElementById('downloadStatus');
        downloadStatus.textContent = '✓ File downloaded successfully!';
        downloadStatus.style.display = 'block';
        setTimeout(() => {
            downloadStatus.style.display = 'none';
        }, 3000);

    } catch (error) {
        console.error('Download error:', error);
        alert('Failed to download file. Please try again.');
    }
});