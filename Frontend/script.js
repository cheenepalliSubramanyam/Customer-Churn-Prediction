document.getElementById('churnForm').addEventListener('submit', async (e) => {
  e.preventDefault();

  const data = {
    MonthlyCharges: parseFloat(document.getElementById('monthlyCharges').value),
    TotalCharges: parseFloat(document.getElementById('totalCharges').value),
    InternetService: document.getElementById('internetService').value,
    tenure: parseFloat(document.getElementById('tenure').value),
    Contract: document.getElementById('contract').value
  };

  try {
    const response = await fetch('https://fjorq3lcze.execute-api.ap-south-1.amazonaws.com/predict_churn', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    const result = await response.json();
    document.getElementById('result').textContent = result.prediction;
  } catch (error) {
    document.getElementById('result').textContent = "Error: " + error;
  }
});
