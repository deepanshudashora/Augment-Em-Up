async function handleFileUpload(event) {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/upload_text', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    document.getElementById('textContent').value = data.text;
}

async function updateText() {
    const text = document.getElementById('textContent').value;
    const formData = new FormData();
    formData.append('text', text);

    const response = await fetch('/update_text', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    alert('Text updated successfully!');
}

async function applyPreprocessing() {
    const checkboxes = document.getElementsByName('preprocessing');
    const selectedSteps = Array.from(checkboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);

    if (selectedSteps.length === 0) {
        alert('Please select at least one preprocessing step.');
        return;
    }

    const formData = new FormData();
    selectedSteps.forEach(step => formData.append('steps', step));

    try {
        const response = await fetch('/apply_preprocessing', {
            method: 'POST',
            body: formData
        });

        const results = await response.json();
        
        // Show the results container
        const resultsContainer = document.querySelector('.results-container');
        if (resultsContainer) {
            resultsContainer.style.display = 'block';
        }

        // Display step-wise results
        const resultsDiv = document.getElementById('preprocessing-results');
        resultsDiv.innerHTML = '';
        
        for (const [step, result] of Object.entries(results.step_results)) {
            const resultElement = document.createElement('div');
            resultElement.innerHTML = `
                <h4>${step.replace('_', ' ').toUpperCase()}:</h4>
                <div class="text-box">${Array.isArray(result) ? result.join(' ') : result}</div>
            `;
            resultsDiv.appendChild(resultElement);
        }

        // Display merged result
        const mergedDiv = document.getElementById('merged-preprocessing-result');
        if (mergedDiv) {
            mergedDiv.textContent = results.merged_result;
        }
    } catch (error) {
        console.error('Error applying preprocessing:', error);
        alert('Error applying preprocessing. Please try again.');
    }
}

function displayPreprocessingResults(results) {
    const resultsDiv = document.getElementById('preprocessing-results');
    const mergedDiv = document.getElementById('merged-preprocessing-result');
    
    if (!results.step_results) return; // Guard clause for old data format
    
    resultsDiv.innerHTML = '';
    
    // Display step-wise results
    for (const [step, result] of Object.entries(results.step_results)) {
        const resultElement = document.createElement('div');
        resultElement.innerHTML = `
            <h4>${step.replace('_', ' ').toUpperCase()}:</h4>
            <div class="text-box">${Array.isArray(result) ? result.join(' ') : result}</div>
        `;
        resultsDiv.appendChild(resultElement);
    }
    
    // Display merged result
    if (mergedDiv) {
        mergedDiv.textContent = results.merged_result;
    }
}

async function applyAugmentation() {
    const checkboxes = document.getElementsByName('augmentation');
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
        const response = await fetch('/apply_augmentation', {
            method: 'POST',
            body: formData
        });

        const results = await response.json();
        
        // Show the results container
        const resultsContainer = document.querySelector('.parallel-results');
        if (resultsContainer) {
            resultsContainer.style.display = 'grid';
        }

        // Display original text results
        const originalResultsDiv = document.getElementById('original-augmentation-results');
        originalResultsDiv.innerHTML = '';
        
        for (const [technique, result] of Object.entries(results.original)) {
            const resultElement = document.createElement('div');
            resultElement.innerHTML = `
                <h4>${technique.replace('_', ' ').toUpperCase()}:</h4>
                <div class="text-box">${result}</div>
            `;
            originalResultsDiv.appendChild(resultElement);
        }

        // Display preprocessed text results
        const preprocessedResultsDiv = document.getElementById('preprocessed-augmentation-results');
        preprocessedResultsDiv.innerHTML = '';
        
        for (const [technique, result] of Object.entries(results.preprocessed)) {
            const resultElement = document.createElement('div');
            resultElement.innerHTML = `
                <h4>${technique.replace('_', ' ').toUpperCase()}:</h4>
                <div class="text-box">${result}</div>
            `;
            preprocessedResultsDiv.appendChild(resultElement);
        }
    } catch (error) {
        console.error('Error applying augmentation:', error);
        alert('Error applying augmentation. Please try again.');
    }
}

function displayAugmentationResults(results) {
    const originalResultsDiv = document.getElementById('original-augmentation-results');
    const preprocessedResultsDiv = document.getElementById('preprocessed-augmentation-results');
    
    originalResultsDiv.innerHTML = '';
    preprocessedResultsDiv.innerHTML = '';
    
    // Display original text augmentations
    for (const [technique, result] of Object.entries(results.original)) {
        const resultElement = document.createElement('div');
        resultElement.innerHTML = `
            <h4>${technique.replace('_', ' ').toUpperCase()}:</h4>
            <div class="text-box">${result}</div>
        `;
        originalResultsDiv.appendChild(resultElement);
    }
    
    // Display preprocessed text augmentations
    for (const [technique, result] of Object.entries(results.preprocessed)) {
        const resultElement = document.createElement('div');
        resultElement.innerHTML = `
            <h4>${technique.replace('_', ' ').toUpperCase()}:</h4>
            <div class="text-box">${result}</div>
        `;
        preprocessedResultsDiv.appendChild(resultElement);
    }
}

function navigateToAugment() {
    // Store current preprocessing state in localStorage before navigation
    const checkboxes = document.getElementsByName('preprocessing');
    const selectedSteps = Array.from(checkboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);
    
    localStorage.setItem('preprocessingSteps', JSON.stringify(selectedSteps));
    window.location.href = '/augment';
}

function clearStorageAndGoHome() {
    localStorage.clear();
    window.location.href = '/';
}

document.addEventListener('DOMContentLoaded', function() {
    // Handle "Select All" for preprocessing
    const selectAllPreprocessing = document.getElementById('select-all-preprocessing');
    if (selectAllPreprocessing) {
        const preprocessingCheckboxes = document.getElementsByName('preprocessing');
        
        // Set default selections if no previous state exists
        if (!localStorage.getItem('preprocessingSteps')) {
            selectAllPreprocessing.checked = true;
            preprocessingCheckboxes.forEach(checkbox => {
                checkbox.checked = true;
            });
        }

        selectAllPreprocessing.addEventListener('change', function() {
            preprocessingCheckboxes.forEach(checkbox => {
                checkbox.checked = this.checked;
            });
        });

        // Update "Select All" checkbox based on individual selections
        preprocessingCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', function() {
                selectAllPreprocessing.checked = 
                    Array.from(preprocessingCheckboxes).every(cb => cb.checked);
            });
        });
    }

    // Handle "Select All" for augmentation
    const selectAllAugmentation = document.getElementById('select-all-augmentation');
    if (selectAllAugmentation) {
        selectAllAugmentation.addEventListener('change', function() {
            const checkboxes = document.getElementsByName('augmentation');
            checkboxes.forEach(checkbox => {
                checkbox.checked = this.checked;
            });
        });
    }

    // Restore preprocessing results if they exist
    const preprocessingResults = document.getElementById('preprocessing-results');
    const mergedResult = document.getElementById('merged-preprocessing-result');
    
    if (preprocessingResults && !preprocessingResults.children.length) {
        // If we're returning to preprocessing page and results exist in storage
        const storedResults = localStorage.getItem('preprocessingResults');
        const storedMerged = localStorage.getItem('mergedResult');
        
        if (storedResults) {
            displayPreprocessingResults(JSON.parse(storedResults));
        }
        
        if (storedMerged && mergedResult) {
            mergedResult.textContent = storedMerged;
        }
    }
}); 