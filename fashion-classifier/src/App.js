import React, { useState } from 'react';
import './App.css';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false); // For loading animation

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setPrediction(null);
    setError(null);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true); 

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPrediction(response.data);
    } catch (err) {
      setError('An error occurred while processing your request.');
      console.error(err);
    } finally {
      setIsLoading(false); 
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Fashion Image Classifier</h1>
      </header>

      <main>
        <form onSubmit={handleSubmit}>
          <input type="file" accept="image/*" onChange={handleFileChange} />
          <button type="submit" disabled={!selectedFile}>
            Predict
          </button>
        </form>

        {error && <div className="error">{error}</div>}

        <div className="preview-area"> 
          {selectedFile && (
            <div>
              <h2>Uploaded Image:</h2>
              <img 
                src={URL.createObjectURL(selectedFile)} 
                alt="Uploaded Preview" 
                className="uploaded-image" 
              />
            </div>
          )}

          {isLoading && (
            <div className="loading-animation">
              {/* You can add a more visually appealing animation here */}
              <div className="spinner"></div> 
              <p>Analyzing Image...</p>
            </div>
          )}

          {prediction && (
            <div className="prediction-results">
              <h2>Prediction Results:</h2>
              <p className="predicted-class">
                <strong>Class:</strong> {prediction.predicted_class}
              </p>
              <p className="confidence">
                <strong>Confidence:</strong> {Math.round(prediction.confidence * 100)}%
              </p> 
            </div>
          )}
        </div> {/* End .preview-area */}
      </main>

      <footer className="App-footer">
        <p>Developed by: [Your Name] </p>
        <p>Contact: [Your Email/Website] </p>
      </footer>
    </div>
  );
}

export default App;
