async function getPredictions() {
    const loading = document.getElementById('loading');
    const predictionsDiv = document.getElementById('predictions');

    loading.innerText = 'Fetching predictions...';

    try {
        const response = await fetch('http://127.0.0.1:5000/predict');
        const data = await response.json();

        console.log('API Response:', data);

        predictionsDiv.innerHTML = ''; // Clear previous content

        if (data && Array.isArray(data.prediction)) {
            const today = new Date();

            data.prediction.forEach((item, index) => {
                const dayDiv = document.createElement('div');
                dayDiv.className = 'day-block';

                const nextDate = new Date(today);
                nextDate.setDate(today.getDate() + index);

                const formattedDate = nextDate.toLocaleDateString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric'
                });

                dayDiv.innerHTML = `
                    <strong>${formattedDate}</strong>
                    <p>Prediction: ${item['Mean Prediction']}</p>
                    <p>Actual Forecast: ${item['Actual Mean']}</p>
                `;

                predictionsDiv.appendChild(dayDiv);
            });
        } else {
            predictionsDiv.innerHTML = `<p>No predictions received from the server.</p>`;
        }
    } catch (error) {
        predictionsDiv.innerHTML = `<p>Error fetching predictions. Try again!</p>`;
        console.error('API Error:', error);
    }
}

// Automatically call getPredictions on page load
window.addEventListener('load', getPredictions);
