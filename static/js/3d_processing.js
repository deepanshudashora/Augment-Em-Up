/**
 * 3D Processing Documentation
 * 
 * This module handles 3D model processing including upload, preprocessing,
 * and augmentation functionalities.
 */

/**
 * Initializes a 3D viewer in the specified container
 * @param {string} containerId - ID of the container element
 * @returns {Object} Scene, camera, and renderer objects
 */
function initViewer(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return null;

    // Clear existing content
    container.innerHTML = '';

    // Setup scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    // Setup camera
    const width = container.clientWidth;
    const height = container.clientHeight || 400;
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(0, 0, 5);

    // Setup renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    // Add controls
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    return { scene, camera, renderer, controls };
}

/**
 * Handles 3D model file upload
 * @param {Event} event - File input change event
 * @returns {Promise<void>}
 */
async function handle3DUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const validExtensions = ['.obj', '.stl', '.ply', '.fbx', '.off'];
    const extension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
    
    if (!validExtensions.includes(extension)) {
        showNotification('error', 'Please upload a valid 3D model file.');
        return;
    }

    const nextButton = document.getElementById('nextButton');
    nextButton.disabled = true;
    nextButton.textContent = 'Uploading...';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/upload_3d', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Upload failed');
        }

        const data = await response.json();

        if (data.success) {
            // Update model info
            updateModelInfo({
                name: file.name,
                size: file.size,
                format: data.statistics.format,
                vertices: data.statistics.vertices,
                faces: data.statistics.faces
            });

            // Display the model
            if (data.model_data) {
                displayModel(data.model_data, '3dPreview');
            }

            // Enable next button
            nextButton.disabled = false;
            nextButton.textContent = 'Next';
            
            showNotification('success', 'File uploaded successfully!');
        } else {
            throw new Error(data.message || 'Upload failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('error', error.message || 'Upload failed. Please try again.');
        nextButton.textContent = 'Next';
        nextButton.disabled = true;
    }
}

/**
 * Applies selected preprocessing steps to the 3D model
 * @returns {Promise<void>}
 * 
 * Available preprocessing steps:
 * - normalize_scale: Scales model to unit cube
 * - center_model: Centers model at origin
 * - remove_duplicates: Removes duplicate vertices
 * - fix_normals: Fixes face winding and normals
 * - simplify_mesh: Reduces mesh complexity
 * - smooth_surface: Applies Laplacian smoothing
 */
async function apply3DPreprocessing() {
    const checkboxes = document.getElementsByName('3d_preprocessing');
    const selectedSteps = Array.from(checkboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);

    if (selectedSteps.length === 0) {
        showNotification('error', 'Please select at least one preprocessing step.');
        return;
    }

    const preprocessButton = document.getElementById('applyPreprocessingBtn');
    const resultsSection = document.getElementById('resultsSection');
    
    preprocessButton.disabled = true;
    preprocessButton.textContent = 'Processing...';

    try {
        const formData = new FormData();
        selectedSteps.forEach(step => formData.append('steps', step));

        const response = await fetch('/apply_3d_preprocessing', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Preprocessing failed');
        }

        const data = await response.json();

        if (data.success) {
            // Show results section
            resultsSection.style.display = 'block';

            // Display stepwise results
            const stepwiseResults = document.getElementById('stepwiseResults');
            stepwiseResults.innerHTML = '';

            for (const [step, result] of Object.entries(data.stepwise_results)) {
                const resultDiv = document.createElement('div');
                resultDiv.className = 'preprocessing-result';
                
                let resultHtml = `
                    <h4>${step.replace(/_/g, ' ').toUpperCase()}</h4>
                    <div class="step-stats">
                `;
                
                if (result.error) {
                    resultHtml += `<p class="error">Error: ${result.error}</p>`;
                } else {
                    resultHtml += `
                        <p>Original: ${result.original_stats.vertices.toLocaleString()} vertices, 
                           ${result.original_stats.faces.toLocaleString()} faces</p>
                        <p>Processed: ${result.processed_stats.vertices.toLocaleString()} vertices, 
                           ${result.processed_stats.faces.toLocaleString()} faces</p>
                        <p class="changes">${result.changes}</p>
                    `;

                    if (result.model_data) {
                        resultHtml += `<div class="step-preview" id="${step}_preview"></div>`;
                    }
                }
                
                resultHtml += `</div>`;
                resultDiv.innerHTML = resultHtml;
                stepwiseResults.appendChild(resultDiv);

                // Display 3D model for this step if available
                if (result.model_data && !result.error) {
                    displayModel(result.model_data, `${step}_preview`);
                }
            }

            // Display final model
            if (data.final_model_data) {
                displayModel(data.final_model_data, 'finalModelViewer');
                
                // Update final stats
                document.getElementById('finalStats').innerHTML = `
                    <p>Final Model Statistics:</p>
                    <p>Vertices: ${data.final_statistics.vertices.toLocaleString()}</p>
                    <p>Faces: ${data.final_statistics.faces.toLocaleString()}</p>
                `;
            }

            // Enable next button
            document.getElementById('nextButton').disabled = false;
            
            showNotification('success', 'Preprocessing completed successfully!');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('error', error.message || 'Preprocessing failed. Please try again.');
    } finally {
        preprocessButton.disabled = false;
        preprocessButton.textContent = 'Apply Preprocessing';
    }
}

/**
 * Applies selected augmentation techniques to the 3D model
 * @returns {Promise<void>}
 * 
 * Available augmentation techniques:
 * - rotate: Random rotation
 * - scale: Random non-uniform scaling
 * - translate: Random translation
 * - noise: Add vertex noise
 * - deform: Sinusoidal deformation
 * - subdivide: Mesh subdivision
 */
async function apply3DAugmentation() {
    const checkboxes = document.getElementsByName('3d_augmentation');
    const selectedAugmentations = Array.from(checkboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);

    if (selectedAugmentations.length === 0) {
        showNotification('error', 'Please select at least one augmentation option.');
        return;
    }

    const augmentButton = document.getElementById('applyAugmentationBtn');
    const resultsSection = document.getElementById('resultsSection');
    
    augmentButton.disabled = true;
    augmentButton.textContent = 'Applying...';

    try {
        const formData = new FormData();
        selectedAugmentations.forEach(aug => formData.append('augmentations[]', aug));

        const response = await fetch('/apply_3d_augmentation', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Augmentation failed');
        }

        const data = await response.json();

        if (data.success) {
            resultsSection.style.display = 'block';

            // Display original model results
            const originalResults = document.getElementById('originalResults');
            originalResults.innerHTML = '';
            
            Object.entries(data.original_results).forEach(([option, result]) => {
                const resultDiv = document.createElement('div');
                resultDiv.className = 'result-item';
                resultDiv.innerHTML = `
                    <h4>${option.replace('_', ' ').toUpperCase()}</h4>
                    <div class="preview-container" id="original_${option}_preview"></div>
                    <p class="result-description">${result.description || ''}</p>
                `;
                originalResults.appendChild(resultDiv);

                if (result.model_data) {
                    displayModel(result.model_data, `original_${option}_preview`);
                }
            });

            // Display preprocessed model results
            const preprocessedResults = document.getElementById('preprocessedResults');
            preprocessedResults.innerHTML = '';
            
            Object.entries(data.preprocessed_results).forEach(([option, result]) => {
                const resultDiv = document.createElement('div');
                resultDiv.className = 'result-item';
                resultDiv.innerHTML = `
                    <h4>${option.replace('_', ' ').toUpperCase()}</h4>
                    <div class="preview-container" id="preprocessed_${option}_preview"></div>
                    <p class="result-description">${result.description || ''}</p>
                `;
                preprocessedResults.appendChild(resultDiv);

                if (result.model_data) {
                    displayModel(result.model_data, `preprocessed_${option}_preview`);
                }
            });

            showNotification('success', 'Augmentation completed successfully!');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('error', 'Augmentation failed. Please try again.');
    } finally {
        augmentButton.disabled = false;
        augmentButton.textContent = 'Apply Augmentation';
    }
}

/**
 * Helper function to format file sizes
 * @param {number} bytes - Size in bytes
 * @returns {string} Formatted size string
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Counts vertices in a 3D model
 * @param {THREE.Object3D} model - Three.js model object
 * @returns {number} Total vertex count
 */
function countVertices(model) {
    let count = 0;
    model.traverse((child) => {
        if (child.geometry) {
            count += child.geometry.attributes.position.count;
        }
    });
    return count;
}

// Usage Examples:
/**
 * Example: Upload and process a 3D model
 * 
 * 1. Upload model:
 * const fileInput = document.getElementById('3dFile');
 * fileInput.addEventListener('change', handle3DUpload);
 * 
 * 2. Preprocess model:
 * await apply3DPreprocessing(['normalize_scale', 'center_model']);
 * 
 * 3. Augment model:
 * await apply3DAugmentation(['rotate', 'scale']);
 */

/**
 * Error Handling Example:
 * 
 * try {
 *     await apply3DPreprocessing();
 * } catch (error) {
 *     processingManager.showError(error.message);
 * }
 */

/**
 * Progress Tracking Example:
 * 
 * processingManager.showLoading('container', 'Processing model...');
 * processingManager.updateProgress(50);
 * // ... processing ...
 * processingManager.hideLoading('container');
 */

// Add notification function
function showNotification(type, message) {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => notification.remove(), 5000);
}

// Helper functions to update UI
function updateModelInfo(info) {
    const elements = {
        modelName: `File: ${info.name}`,
        modelSize: `Size: ${formatFileSize(info.size)}`,
        modelFormat: `Format: ${info.format}`,
        modelVertices: `Vertices: ${info.vertices.toLocaleString()}`,
        modelFaces: `Faces: ${info.faces.toLocaleString()}`
    };

    for (const [id, text] of Object.entries(elements)) {
        const element = document.getElementById(id);
        if (element) element.textContent = text;
    }
}

function enableNextButton() {
    const nextButton = document.getElementById('nextButton');
    if (nextButton) {
        nextButton.disabled = false;
        nextButton.textContent = 'Next';
    }
}

function disableNextButton() {
    const nextButton = document.getElementById('nextButton');
    if (nextButton) {
        nextButton.disabled = true;
        nextButton.textContent = 'Next';
    }
}

// Global variables
let currentViewer = null;
let currentModel = null;

// Load and display model
function displayModel(modelData, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Clear existing content
    while (container.firstChild) {
        container.removeChild(container.firstChild);
    }

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Setup renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    // Setup scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    // Setup camera
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(0, 0, 5);

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    // Setup controls
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // Load model
    const loader = new THREE.OBJLoader();
    const model = loader.parse(atob(modelData));
    
    // Center and scale model
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const scale = 2 / maxDim;
    
    model.scale.multiplyScalar(scale);
    model.position.sub(center.multiplyScalar(scale));
    scene.add(model);

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();
}

function displayPreprocessingResults(results) {
    const container = document.getElementById('preprocessing-results');
    if (!container) return;

    container.innerHTML = '<h3>Preprocessing Results:</h3>';
    
    for (const [step, result] of Object.entries(results)) {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'preprocessing-result';
        
        let resultText = `<strong>${step.replace('_', ' ').toUpperCase()}</strong><br>`;
        
        if (result.error) {
            resultText += `<span class="error">Error: ${result.error}</span>`;
        } else {
            for (const [key, value] of Object.entries(result)) {
                resultText += `${key.replace('_', ' ')}: ${formatValue(value)}<br>`;
            }
        }
        
        resultDiv.innerHTML = resultText;
        container.appendChild(resultDiv);
    }
}

function formatValue(value) {
    if (Array.isArray(value)) {
        return value.map(v => typeof v === 'number' ? v.toFixed(3) : v).join(', ');
    }
    if (typeof value === 'number') {
        return value.toFixed(3);
    }
    return value;
}

// Add this function
function initializePreprocessPage() {
    // Get model data from the server
    fetch('/get_model_data')
        .then(response => response.json())
        .then(data => {
            if (data.model_data) {
                // Display the input model
                displayModel(data.model_data, 'inputModelViewer');
                
                // Update model stats
                const statsDiv = document.getElementById('modelStats');
                statsDiv.innerHTML = `
                    Vertices: ${data.stats.vertices.toLocaleString()}<br>
                    Faces: ${data.stats.faces.toLocaleString()}
                `;
            }
        })
        .catch(error => {
            console.error('Error loading model:', error);
            showNotification('error', 'Error loading model');
        });
}

// Call this when the page loads
document.addEventListener('DOMContentLoaded', initializePreprocessPage);

// Add select all functionality
document.addEventListener('DOMContentLoaded', function() {
    const selectAllCheckbox = document.getElementById('selectAll');
    const optionCheckboxes = document.getElementsByName('3d_preprocessing');

    // Handle select all changes
    selectAllCheckbox.addEventListener('change', function() {
        optionCheckboxes.forEach(checkbox => {
            checkbox.checked = this.checked;
        });
    });

    // Handle individual checkbox changes
    optionCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            selectAllCheckbox.checked = Array.from(optionCheckboxes).every(cb => cb.checked);
        });
    });

    // Initialize the page
    initializePreprocessPage();
});

// Add this function to initialize the page
document.addEventListener('DOMContentLoaded', function() {
    initializeAugmentPage();

    // Handle select all functionality
    const selectAllCheckbox = document.getElementById('selectAll');
    const optionCheckboxes = document.getElementsByName('3d_augmentation');

    selectAllCheckbox.addEventListener('change', function() {
        optionCheckboxes.forEach(checkbox => {
            checkbox.checked = this.checked;
        });
    });

    optionCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            selectAllCheckbox.checked = Array.from(optionCheckboxes).every(cb => cb.checked);
        });
    });
});

async function initializeAugmentPage() {
    try {
        const response = await fetch('/get_model_data');
        if (!response.ok) {
            const errorDetail = await response.json();
            throw new Error(errorDetail.detail || 'Failed to load model data');
        }
        const data = await response.json();
        if (data.model_data) {
            displayModel(data.model_data, 'inputModelViewer');
        } else {
            throw new Error('Model data is not available');
        }
    } catch (error) {
        console.error('Error loading model:', error);
        showNotification('error', error.message || 'Error loading model');
    }
}