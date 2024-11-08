let wavesurfer;

document.addEventListener('DOMContentLoaded', function() {
    // Initialize WaveSurfer
    wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#4a9eff',
        progressColor: '#1e88e5',
        cursorColor: '#fff',
        barWidth: 2,
        barRadius: 3,
        cursorWidth: 1,
        height: 100,
        barGap: 3
    });

    // Initialize file input listener
    const audioFileInput = document.getElementById('audioFile');
    if (audioFileInput) {
        audioFileInput.addEventListener('change', handleAudioUpload);
    }

    // Add preprocessing button listener
    const preprocessingBtn = document.getElementById('applyPreprocessingBtn');
    if (preprocessingBtn) {
        preprocessingBtn.addEventListener('click', applyAudioPreprocessing);
    }

    // Check if we're on the preprocessing page
    const nextButton = document.getElementById('nextButton');
    if (nextButton) {
        // Check if preprocessing was applied
        const preprocessingApplied = localStorage.getItem('preprocessingApplied') === 'true';
        nextButton.disabled = !preprocessingApplied;
    }
});

async function handleAudioUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Show loading state
    const nextButton = document.getElementById('nextButton');
    nextButton.disabled = true;
    nextButton.textContent = 'Uploading...';

    // Update audio info
    document.getElementById('audioName').textContent = `File: ${file.name}`;
    document.getElementById('audioSize').textContent = `Size: ${formatFileSize(file.size)}`;

    // Create object URL for audio preview
    const audioURL = URL.createObjectURL(file);
    const audioPreview = document.getElementById('audioPreview');
    audioPreview.src = audioURL;

    // Load audio into WaveSurfer
    wavesurfer.load(audioURL);

    // Get audio duration after loading
    wavesurfer.on('ready', function() {
        const duration = wavesurfer.getDuration();
        document.getElementById('audioDuration').textContent = 
            `Duration: ${formatDuration(duration)}`;
    });

    // Upload to server
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload_audio', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Upload response:', data); // Debug log
        
        if (data.success) {
            // Enable next button and update text
            nextButton.disabled = false;
            nextButton.textContent = 'Next';
            
            // Store upload state
            localStorage.setItem('audioUploaded', 'true');
        } else {
            throw new Error('Upload failed');
        }
    } catch (error) {
        console.error('Error uploading audio:', error);
        alert('Error uploading audio. Please try again.');
        nextButton.textContent = 'Next';
    }
}

async function applyAudioPreprocessing() {
    console.log('Starting preprocessing...'); // Debug log

    const checkboxes = document.getElementsByName('audio_preprocessing');
    const selectedSteps = Array.from(checkboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);

    if (selectedSteps.length === 0) {
        alert('Please select at least one preprocessing step.');
        return;
    }

    // Show loading state
    const preprocessButton = document.getElementById('applyPreprocessingBtn');
    const originalButtonText = preprocessButton.textContent;
    preprocessButton.textContent = 'Processing...';
    preprocessButton.disabled = true;

    try {
        console.log('Selected steps:', selectedSteps); // Debug log

        const formData = new FormData();
        selectedSteps.forEach(step => formData.append('steps', step));

        const response = await fetch('/apply_audio_preprocessing', {
            method: 'POST',
            body: formData
        });

        console.log('Response status:', response.status); // Debug log

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const results = await response.json();
        console.log('Preprocessing results:', results); // Debug log

        if (results.success) {
            // Display step-wise results
            const resultsDiv = document.getElementById('preprocessing-results');
            resultsDiv.innerHTML = '';
            
            Object.entries(results.step_results).forEach(([step, data]) => {
                const resultElement = document.createElement('div');
                resultElement.className = 'audio-result';
                resultElement.innerHTML = `
                    <h4>${step.replace('_', ' ').toUpperCase()}</h4>
                    <audio controls>
                        <source src="data:audio/wav;base64,${data.audio}" type="audio/wav">
                    </audio>
                    <div class="waveform" id="waveform-${step}"></div>
                    <div class="audio-info">
                        <p>Duration: ${data.duration}</p>
                        <p>Sample Rate: ${data.sample_rate} Hz</p>
                    </div>
                `;
                resultsDiv.appendChild(resultElement);

                // Create waveform after element is added to DOM
                setTimeout(() => {
                    createWaveform(`waveform-${step}`, `data:audio/wav;base64,${data.audio}`);
                }, 0);
            });

            // Display merged result
            const mergedDiv = document.getElementById('merged-preprocessing-result');
            mergedDiv.innerHTML = `
                <div class="audio-result">
                    <h4>FINAL RESULT</h4>
                    <audio controls>
                        <source src="data:audio/wav;base64,${results.merged_result.audio}" type="audio/wav">
                    </audio>
                    <div class="waveform" id="waveform-merged"></div>
                    <div class="audio-info">
                        <p>Duration: ${results.merged_result.duration}</p>
                        <p>Sample Rate: ${results.merged_result.sample_rate} Hz</p>
                    </div>
                </div>
            `;

            // Create waveform for merged result
            setTimeout(() => {
                createWaveform('waveform-merged', `data:audio/wav;base64,${results.merged_result.audio}`);
            }, 0);

            // Enable next button
            const nextButton = document.getElementById('nextButton');
            if (nextButton) {
                nextButton.disabled = false;
                console.log('Next button enabled'); // Debug log
            }

            // Store preprocessing state
            localStorage.setItem('preprocessingApplied', 'true');
        } else {
            throw new Error('Preprocessing failed');
        }
    } catch (error) {
        console.error('Error applying audio preprocessing:', error);
        alert('Error applying audio preprocessing. Please try again.');
        preprocessButton.textContent = originalButtonText;
        preprocessButton.disabled = false;
    } finally {
        // Reset preprocessing button state
        preprocessButton.textContent = originalButtonText;
        preprocessButton.disabled = false;
    }
}

async function applyAudioAugmentation() {
    console.log('Starting augmentation...'); // Debug log

    const checkboxes = document.getElementsByName('audio_augmentation');
    const selectedTechniques = Array.from(checkboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);

    if (selectedTechniques.length === 0) {
        alert('Please select at least one augmentation technique.');
        return;
    }

    // Show loading state
    const augmentButton = document.getElementById('applyAugmentationBtn');
    const originalButtonText = augmentButton.textContent;
    augmentButton.textContent = 'Processing...';
    augmentButton.disabled = true;

    try {
        console.log('Selected techniques:', selectedTechniques); // Debug log

        const formData = new FormData();
        selectedTechniques.forEach(technique => formData.append('techniques', technique));

        const response = await fetch('/apply_audio_augmentation', {
            method: 'POST',
            body: formData
        });

        console.log('Response status:', response.status); // Debug log

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const results = await response.json();
        console.log('Augmentation results:', results); // Debug log

        if (results.success) {
            // Display original audio augmentations
            const originalResultsDiv = document.getElementById('original-augmentation-results');
            originalResultsDiv.innerHTML = '<h3>Original Audio Augmentations:</h3>';
            
            Object.entries(results.original_results).forEach(([technique, data]) => {
                const resultElement = document.createElement('div');
                resultElement.className = 'audio-result';
                resultElement.innerHTML = `
                    <h4>${technique.replace('_', ' ').toUpperCase()}</h4>
                    <p class="technique-params">${data.parameters}</p>
                    <audio controls>
                        <source src="data:audio/wav;base64,${data.audio}" type="audio/wav">
                    </audio>
                    <div class="waveform" id="waveform-original-${technique}"></div>
                    <div class="audio-info">
                        <p>Duration: ${data.duration}</p>
                        <p>Sample Rate: ${data.sample_rate} Hz</p>
                    </div>
                `;
                originalResultsDiv.appendChild(resultElement);

                // Create waveform after element is added to DOM
                setTimeout(() => {
                    createWaveform(`waveform-original-${technique}`, 
                                 `data:audio/wav;base64,${data.audio}`);
                }, 0);
            });

            // Display preprocessed audio augmentations
            const preprocessedResultsDiv = document.getElementById('preprocessed-augmentation-results');
            preprocessedResultsDiv.innerHTML = '<h3>Preprocessed Audio Augmentations:</h3>';
            
            Object.entries(results.preprocessed_results).forEach(([technique, data]) => {
                const resultElement = document.createElement('div');
                resultElement.className = 'audio-result';
                resultElement.innerHTML = `
                    <h4>${technique.replace('_', ' ').toUpperCase()}</h4>
                    <p class="technique-params">${data.parameters}</p>
                    <audio controls>
                        <source src="data:audio/wav;base64,${data.audio}" type="audio/wav">
                    </audio>
                    <div class="waveform" id="waveform-preprocessed-${technique}"></div>
                    <div class="audio-info">
                        <p>Duration: ${data.duration}</p>
                        <p>Sample Rate: ${data.sample_rate} Hz</p>
                    </div>
                `;
                preprocessedResultsDiv.appendChild(resultElement);

                // Create waveform after element is added to DOM
                setTimeout(() => {
                    createWaveform(`waveform-preprocessed-${technique}`, 
                                 `data:audio/wav;base64,${data.audio}`);
                }, 0);
            });
        }

    } catch (error) {
        console.error('Error applying augmentation:', error);
        alert('Error applying augmentation. Please try again.');
        augmentButton.textContent = originalButtonText;
        augmentButton.disabled = false;
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(seconds) {
    const minutes = Math.floor(seconds / 60);
    seconds = Math.floor(seconds % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function navigateToPreprocess() {
    window.location.href = '/audio_preprocess';
}

function navigateToAugment() {
    window.location.href = '/audio_augment';
}