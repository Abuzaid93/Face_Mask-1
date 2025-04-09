document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const selectedFile = document.getElementById('selected-file');
    const loading = document.getElementById('loading');
    const resultContainer = document.getElementById('result-container');
    const errorContainer = document.getElementById('error-container');
    const resultImage = document.getElementById('result-image');
    const resultStatus = document.getElementById('result-status');
    const resultText = document.getElementById('result-text');
    const accuracyBar = document.getElementById('accuracy-bar');
    const accuracyValue = document.getElementById('accuracy-value');
    const tryAgainBtn = document.getElementById('try-again-btn');
    const errorTryAgainBtn = document.getElementById('error-try-again-btn');
    const errorMessage = document.getElementById('error-message');

    // File selection change
    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            selectedFile.textContent = this.files[0].name;
        } else {
            selectedFile.textContent = '';
        }
    });

    // Form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!fileInput.files || !fileInput.files[0]) {
            showError('Please select an image file');
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        // Show loading
        showLoading();
        
        // Send request to server
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.error) {
                showError(data.error);
                return;
            }
            
            // Display result
            resultImage.src = 'data:image/jpeg;base64,' + data.image;
            resultText.textContent = data.result;
            
            // Set status class and text
            resultStatus.textContent = data.status === 'with_mask' ? 'Mask Detected' : 'No Mask Detected';
            resultStatus.className = 'result-status ' + data.status;
            
            // Update confidence bar
            const confidencePercent = (data.probability * 100).toFixed(2);
            accuracyBar.style.width = confidencePercent + '%';
            accuracyValue.textContent = confidencePercent + '%';
            
            // Show result container
            resultContainer.classList.remove('hidden');
        })
        .catch(error => {
            hideLoading();
            showError('An error occurred: ' + error.message);
        });
    });

    // Try again buttons
    tryAgainBtn.addEventListener('click', resetForm);
    errorTryAgainBtn.addEventListener('click', resetForm);

    function showLoading() {
        loading.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        errorContainer.classList.add('hidden');
    }

    function hideLoading() {
        loading.classList.add('hidden');
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorContainer.classList.remove('hidden');
        resultContainer.classList.add('hidden');
    }

    function resetForm() {
        uploadForm.reset();
        selectedFile.textContent = '';
        resultContainer.classList.add('hidden');
        errorContainer.classList.add('hidden');
    }
});