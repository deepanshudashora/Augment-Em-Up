async function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Display preview
    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById('originalImage').src = e.target.result;
    }
    reader.readAsDataURL(file);

    // Upload to server
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload_image', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            // Enable the next button
            document.getElementById('nextButton').disabled = false;
        } else {
            throw new Error('Upload failed');
        }
    } catch (error) {
        console.error('Error uploading image:', error);
        alert('Error uploading image. Please try again.');
    }
}

function navigateToPreprocess() {
    window.location.href = '/image_preprocess';
}

// Add this to handle the select all functionality
document.addEventListener('DOMContentLoaded', function() {
    const selectAllPreprocessing = document.getElementById('select-all-preprocessing');
    if (selectAllPreprocessing) {
        const preprocessingCheckboxes = document.getElementsByName('image_preprocessing');
        selectAllPreprocessing.addEventListener('change', function() {
            preprocessingCheckboxes.forEach(checkbox => {
                checkbox.checked = this.checked;
            });
        });
    }
});

// Add this function for navigation to augmentation
function navigateToAugment() {
    window.location.href = '/image_augment';
}

// Update the applyImagePreprocessing function to enable navigation after processing
async function applyImagePreprocessing() {
    const checkboxes = document.getElementsByName('image_preprocessing');
    const selectedSteps = Array.from(checkboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);

    if (selectedSteps.length === 0) {
        alert('Please select at least one preprocessing step.');
        return;
    }

    // Show loading state
    const applyButton = document.querySelector('.preprocessing-options button');
    const originalButtonText = applyButton.textContent;
    applyButton.textContent = 'Processing...';
    applyButton.disabled = true;

    const formData = new FormData();
    selectedSteps.forEach(step => formData.append('steps', step));

    try {
        const response = await fetch('/apply_image_preprocessing', {
            method: 'POST',
            body: formData
        });

        const results = await response.json();
        
        if (!response.ok) {
            throw new Error(results.detail || 'Error applying preprocessing');
        }

        // Display step-wise results
        const resultsDiv = document.getElementById('preprocessing-results');
        resultsDiv.innerHTML = '';
        
        Object.entries(results.step_results).forEach(([step, data]) => {
            const resultElement = document.createElement('div');
            resultElement.className = 'image-result';
            resultElement.innerHTML = `
                <h4>${step.replace('_', ' ').toUpperCase()}</h4>
                <img src="data:image/png;base64,${data}" alt="${step}">
                <p class="image-size">Size: ${results.sizes[step]}</p>
            `;
            resultsDiv.appendChild(resultElement);
        });

        // Display merged result
        const mergedDiv = document.getElementById('merged-preprocessing-result');
        mergedDiv.innerHTML = `
            <div class="image-result">
                <h4>FINAL RESULT</h4>
                <img src="data:image/png;base64,${results.merged_result}" alt="Merged Result">
                <p class="image-size">Size: ${results.merged_size}</p>
            </div>
        `;

        // Show results container and enable next button
        document.querySelector('.parallel-results').style.display = 'grid';
        const nextButton = document.querySelector('.nav-right button');
        if (nextButton) {
            nextButton.disabled = false;
        }

        // Store preprocessing state
        localStorage.setItem('preprocessingApplied', 'true');

    } catch (error) {
        console.error('Error applying preprocessing:', error);
    } finally {
        // Reset button state
        applyButton.textContent = originalButtonText;
        applyButton.disabled = false;
    }
}

// Add this to handle page load state
document.addEventListener('DOMContentLoaded', function() {
    // Check if preprocessing has been applied
    const preprocessingApplied = localStorage.getItem('preprocessingApplied') === 'true';
    const nextButton = document.querySelector('.nav-right button');
    
    if (nextButton) {
        nextButton.disabled = !preprocessingApplied;
    }
});

async function applyImageAugmentation() {
    const checkboxes = document.getElementsByName('image_augmentation');
    const selectedTechniques = Array.from(checkboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);

    if (selectedTechniques.length === 0) {
        alert('Please select at least one augmentation technique.');
        return;
    }

    const formData = new FormData();
    selectedTechniques.forEach(technique => formData.append('techniques', technique));

    try {
        const response = await fetch('/apply_image_augmentation', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const results = await response.json();
        
        if (results.success) {
            // Display original augmentations
            const originalResultsDiv = document.getElementById('original-augmentation-results');
            originalResultsDiv.innerHTML = '';
            
            Object.entries(results.original).forEach(([technique, data]) => {
                const resultElement = document.createElement('div');
                resultElement.className = 'image-result';
                resultElement.innerHTML = `
                    <h4>${technique.replace('_', ' ').toUpperCase()}</h4>
                    <img src="data:image/png;base64,${data.image}" alt="${technique}">
                    <p class="image-size">Size: ${data.size}</p>
                `;
                originalResultsDiv.appendChild(resultElement);
            });

            // Display preprocessed augmentations
            const preprocessedResultsDiv = document.getElementById('preprocessed-augmentation-results');
            preprocessedResultsDiv.innerHTML = '';
            
            Object.entries(results.preprocessed).forEach(([technique, data]) => {
                const resultElement = document.createElement('div');
                resultElement.className = 'image-result';
                resultElement.innerHTML = `
                    <h4>${technique.replace('_', ' ').toUpperCase()}</h4>
                    <img src="data:image/png;base64,${data.image}" alt="${technique}">
                    <p class="image-size">Size: ${data.size}</p>
                `;
                preprocessedResultsDiv.appendChild(resultElement);
            });

            // Show results container
            document.querySelector('.parallel-results').style.display = 'grid';
        }
    } catch (error) {
        console.error('Error applying augmentation:', error);
        alert('Error applying augmentation. Please try again.');
    }
} 