/**
 * Computer Vision Tasks - Operations Page JavaScript
 * Handles image upload, Harris detection, SIFT extraction, and feature matching
 */

// API Configuration
const API_BASE_URL = 'http://localhost:8080';

// Global state
let uploadedImages = [];
let harrisResults = null;
let siftResults = null;

/**
 * Helper: Show toast notification
 */
function showToast(message, type = 'success') {
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        toastContainer.style.zIndex = '9999';
        document.body.appendChild(toastContainer);
    }
    
    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true" data-bs-autohide="true" data-bs-delay="3000">
            <div class="toast-header ${type === 'error' ? 'bg-danger' : 'bg-success'} text-white">
                <i class="bi ${type === 'error' ? 'bi-exclamation-triangle' : 'bi-check-circle'} me-2"></i>
                <strong class="me-auto">${type === 'error' ? 'Error' : 'Success'}</strong>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

/**
 * Helper: Update performance table
 */
function updatePerformanceTable(operation, timeMs, metrics, status = 'completed') {
    const rowId = `perf${operation.charAt(0).toUpperCase() + operation.slice(1)}`;
    const row = document.getElementById(rowId);
    if (!row) return;
    
    const timeCell = row.cells[1];
    const metricsCell = row.cells[2];
    const statusCell = row.cells[3];
    
    if (status === 'running') {
        timeCell.innerHTML = '<span class="spinner-border spinner-border-sm"></span>';
        metricsCell.innerHTML = 'Processing...';
        statusCell.innerHTML = '<span class="badge bg-warning">Running</span>';
    } else if (status === 'completed') {
        timeCell.innerHTML = `<strong>${timeMs.toFixed(2)} ms</strong>`;
        metricsCell.innerHTML = metrics;
        statusCell.innerHTML = '<span class="badge bg-success">Completed</span>';
    } else if (status === 'error') {
        timeCell.innerHTML = '—';
        metricsCell.innerHTML = metrics;
        statusCell.innerHTML = '<span class="badge bg-danger">Error</span>';
    }
}

/**
 * Upload Images to Backend
 */
async function uploadImages(files) {
    const formData = new FormData();
    
    for (let i = 0; i < files.length; i++) {
        formData.append('images', files[i]);
    }
    
    // Show progress bar
    const progressDiv = document.getElementById('uploadProgress');
    progressDiv.style.display = 'block';
    const progressBar = progressDiv.querySelector('.progress-bar');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Upload failed');
        
        const data = await response.json();
        
        if (data.success) {
            uploadedImages = data.images;
            displayImageGallery(uploadedImages);
            showToast(`Successfully uploaded ${data.image_count} image(s)`, 'success');
            
            // Clear previous results
            clearPreviousResults();
        } else {
            throw new Error(data.error || 'Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showToast(error.message, 'error');
    } finally {
        progressDiv.style.display = 'none';
        progressBar.style.width = '0%';
    }
}

/**
 * Display uploaded images in gallery
 */
function displayImageGallery(images) {
    const gallery = document.getElementById('imageGallery');
    const container = document.getElementById('galleryContainer');
    
    if (!images || images.length === 0) {
        gallery.style.display = 'none';
        return;
    }
    
    gallery.style.display = 'block';
    container.innerHTML = '';
    
    images.forEach((img, idx) => {
        const col = document.createElement('div');
        col.className = 'col-md-3 col-sm-4 col-6';
        col.innerHTML = `
            <div class="gallery-item" onclick="openImageModal('${img.preview}', '${img.filename}')">
                <img src="${img.preview}" alt="${img.filename}" loading="lazy">
                <div class="overlay">${img.filename}<br>${img.width}×${img.height}</div>
            </div>
        `;
        container.appendChild(col);
    });
}

/**
 * Clear previous operation results
 */
function clearPreviousResults() {
    // Hide all result containers
    document.getElementById('harrisResults').style.display = 'none';
    document.getElementById('siftResults').style.display = 'none';
    document.getElementById('ssdResults').style.display = 'none';
    document.getElementById('nccResults').style.display = 'none';
    
    // Reset performance table for pending operations
    ['harris', 'sift', 'ssd', 'ncc'].forEach(op => {
        updatePerformanceTable(op, 0, '—', 'pending');
    });
}

/**
 * Run Harris Corner Detection
 */
async function runHarris() {
    if (uploadedImages.length === 0) {
        showToast('Please upload images first', 'error');
        return;
    }
    
    const btn = document.getElementById('runHarrisBtn');
    const progressDiv = document.getElementById('harrisProgress');
    const resultsDiv = document.getElementById('harrisResults');
    
    // UI updates
    btn.disabled = true;
    btn.classList.add('btn-loading');
    progressDiv.style.display = 'flex';
    resultsDiv.style.display = 'none';
    updatePerformanceTable('harris', 0, 'Detecting corners...', 'running');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/harris`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (!response.ok) throw new Error('Harris detection failed');
        
        const data = await response.json();
        
        if (data.success) {
            harrisResults = data;
            
            // Display results
            displayHarrisResults(data);
            
            // Update performance table
            updatePerformanceTable('harris', data.computation_time_ms, `${data.total_corners} corners detected`, 'completed');
            
            showToast(`Harris detection completed in ${data.computation_time_ms.toFixed(2)}ms`, 'success');
        } else {
            throw new Error(data.error || 'Harris detection failed');
        }
    } catch (error) {
        console.error('Harris error:', error);
        showToast(error.message, 'error');
        updatePerformanceTable('harris', 0, 'Error', 'error');
    } finally {
        btn.disabled = false;
        btn.classList.remove('btn-loading');
        progressDiv.style.display = 'none';
    }
}

/**
 * Display Harris detection results
 */
function displayHarrisResults(data) {
    const resultsDiv = document.getElementById('harrisResults');
    const originalContainer = document.getElementById('harrisOriginal');
    const resultContainer = document.getElementById('harrisResult');
    const timeSpan = document.getElementById('harrisTime');
    
    timeSpan.textContent = `${data.computation_time_ms.toFixed(2)} ms`;
    
    // Clear containers
    originalContainer.innerHTML = '';
    resultContainer.innerHTML = '';
    
    // Display each image's results
    uploadedImages.forEach((img, idx) => {
        const resultImg = data.result_images[idx];
        
        // Original image card
        const originalCard = document.createElement('div');
        originalCard.className = 'image-card';
        originalCard.innerHTML = `
            <img src="${img.preview}" alt="Original ${idx + 1}" class="img-fluid rounded">
            <div class="image-caption">Original ${idx + 1}</div>
        `;
        originalContainer.appendChild(originalCard);
        
        // Result image card
        const resultCard = document.createElement('div');
        resultCard.className = 'image-card';
        resultCard.innerHTML = `
            <img src="${resultImg}" alt="Harris ${idx + 1}" class="img-fluid rounded">
            <div class="image-caption">${data.per_image_data[idx]?.corners_detected || 0} corners</div>
        `;
        resultContainer.appendChild(resultCard);
    });
    
    resultsDiv.style.display = 'block';
}

/**
 * Run SIFT Feature Extraction
 */
async function runSIFT() {
    if (uploadedImages.length === 0) {
        showToast('Please upload images first', 'error');
        return;
    }
    
    const btn = document.getElementById('runSiftBtn');
    const progressDiv = document.getElementById('siftProgress');
    const resultsDiv = document.getElementById('siftResults');
    
    btn.disabled = true;
    btn.classList.add('btn-loading');
    progressDiv.style.display = 'flex';
    resultsDiv.style.display = 'none';
    updatePerformanceTable('sift', 0, 'Extracting features...', 'running');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/sift`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (!response.ok) throw new Error('SIFT extraction failed');
        
        const data = await response.json();
        
        if (data.success) {
            siftResults = data;
            displaySIFTResults(data);
            
            updatePerformanceTable('sift', data.computation_time_ms, `${data.total_keypoints} keypoints extracted`, 'completed');
            
            showToast(`SIFT extraction completed in ${data.computation_time_ms.toFixed(2)}ms`, 'success');
        } else {
            throw new Error(data.error || 'SIFT extraction failed');
        }
    } catch (error) {
        console.error('SIFT error:', error);
        showToast(error.message, 'error');
        updatePerformanceTable('sift', 0, 'Error', 'error');
    } finally {
        btn.disabled = false;
        btn.classList.remove('btn-loading');
        progressDiv.style.display = 'none';
    }
}

/**
 * Display SIFT results
 */
function displaySIFTResults(data) {
    const resultsDiv = document.getElementById('siftResults');
    const originalContainer = document.getElementById('siftOriginal');
    const resultContainer = document.getElementById('siftResult');
    const timeSpan = document.getElementById('siftTime');
    
    timeSpan.textContent = `${data.computation_time_ms.toFixed(2)} ms`;
    
    originalContainer.innerHTML = '';
    resultContainer.innerHTML = '';
    
    uploadedImages.forEach((img, idx) => {
        const resultImg = data.result_images[idx];
        
        const originalCard = document.createElement('div');
        originalCard.className = 'image-card';
        originalCard.innerHTML = `
            <img src="${img.preview}" alt="Original ${idx + 1}" class="img-fluid rounded">
            <div class="image-caption">Original ${idx + 1}</div>
        `;
        originalContainer.appendChild(originalCard);
        
        const resultCard = document.createElement('div');
        resultCard.className = 'image-card';
        resultCard.innerHTML = `
            <img src="${resultImg}" alt="SIFT ${idx + 1}" class="img-fluid rounded">
            <div class="image-caption">${data.per_image_data[idx]?.keypoints_detected || 0} keypoints</div>
        `;
        resultContainer.appendChild(resultCard);
    });
    
    resultsDiv.style.display = 'block';
}

/**
 * Run Feature Matching (SSD or NCC)
 */
async function runMatching(type) {
    if (uploadedImages.length < 2) {
        showToast('Need at least 2 images for matching', 'error');
        return;
    }
    
    const endpoint = type === 'ssd' ? '/api/match-ssd' : '/api/match-ncc';
    const progressId = `${type}Progress`;
    const resultsId = `${type}Results`;
    const timeId = `${type}Time`;
    const matchesId = `${type}Matches`;
    const visId = `${type}Visualization`;
    const btnId = `run${type.toUpperCase()}Btn`;
    
    const btn = document.getElementById(btnId);
    const progressDiv = document.getElementById(progressId);
    const resultsDiv = document.getElementById(resultsId);
    
    btn.disabled = true;
    btn.classList.add('btn-loading');
    progressDiv.style.display = 'block';
    resultsDiv.style.display = 'none';
    updatePerformanceTable(type, 0, `Matching with ${type.toUpperCase()}...`, 'running');
    
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (!response.ok) throw new Error(`${type.toUpperCase()} matching failed`);
        
        const data = await response.json();
        
        if (data.success) {
            // Display results
            document.getElementById(timeId).textContent = `${data.computation_time_ms.toFixed(2)} ms`;
            
            const matchText = type === 'ssd' 
                ? `${data.num_matches} matches, avg distance: ${data.avg_distance.toFixed(3)}`
                : `${data.num_matches} matches, avg correlation: ${(data.avg_correlation * 100).toFixed(1)}%`;
            document.getElementById(matchesId).textContent = matchText;
            
            document.getElementById(visId).src = data.visualization;
            resultsDiv.style.display = 'block';
            
            const metrics = type === 'ssd'
                ? `${data.num_matches} matches, ${data.avg_distance.toFixed(3)} avg dist`
                : `${data.num_matches} matches, ${(data.avg_correlation * 100).toFixed(1)}% avg corr`;
            updatePerformanceTable(type, data.computation_time_ms, metrics, 'completed');
            
            showToast(`${type.toUpperCase()} matching completed in ${data.computation_time_ms.toFixed(2)}ms`, 'success');
        } else {
            throw new Error(data.error || `${type.toUpperCase()} matching failed`);
        }
    } catch (error) {
        console.error(`${type} matching error:`, error);
        showToast(error.message, 'error');
        updatePerformanceTable(type, 0, 'Error', 'error');
    } finally {
        btn.disabled = false;
        btn.classList.remove('btn-loading');
        progressDiv.style.display = 'none';
    }
}

/**
 * Download result images
 */
function downloadResults(operation) {
    let images = [];
    
    if (operation === 'harris' && harrisResults) {
        images = harrisResults.result_images;
    } else if (operation === 'sift' && siftResults) {
        images = siftResults.result_images;
    } else {
        showToast('No results to download. Run the operation first.', 'error');
        return;
    }
    
    images.forEach((imgBase64, idx) => {
        const link = document.createElement('a');
        link.download = `${operation}_result_${idx + 1}.jpg`;
        link.href = imgBase64;
        link.click();
    });
    
    showToast(`Downloaded ${images.length} image(s)`, 'success');
}

/**
 * Open image in modal (lightbox)
 */
function openImageModal(imgSrc, title) {
    // Create modal dynamically
    const modalHtml = `
        <div class="modal fade" id="imageModal" tabindex="-1" aria-hidden="true">
            <div class="modal-dialog modal-lg modal-dialog-centered">
                <div class="modal-content bg-dark">
                    <div class="modal-header border-0">
                        <h5 class="modal-title text-white">${title}</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body text-center">
                        <img src="${imgSrc}" alt="${title}" class="img-fluid">
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Remove existing modal if any
    const existingModal = document.getElementById('imageModal');
    if (existingModal) existingModal.remove();
    
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    const modal = new bootstrap.Modal(document.getElementById('imageModal'));
    modal.show();
    
    document.getElementById('imageModal').addEventListener('hidden.bs.modal', () => {
        document.getElementById('imageModal').remove();
    });
}

/**
 * Initialize drag and drop functionality
 */
function initDragAndDrop() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        
        const files = Array.from(e.dataTransfer.files).filter(f => 
            f.type.startsWith('image/')
        );
        
        if (files.length > 0) {
            uploadImages(files);
        } else {
            showToast('Please drop image files only', 'error');
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadImages(Array.from(e.target.files));
        }
    });
}

/**
 * Load default images from backend on page load
 */
async function loadDefaultImages() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/images`);
        if (response.ok) {
            const data = await response.json();
            if (data.success && data.images && data.images.length > 0) {
                uploadedImages = data.images;
                displayImageGallery(uploadedImages);
                showToast(`Loaded ${data.image_count} default image(s)`, 'info');
            }
        }
    } catch (error) {
        console.log('No default images or backend not running');
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initDragAndDrop();
    loadDefaultImages();
    
    // Add CSS for image cards if not present
    const style = document.createElement('style');
    style.textContent = `
        .image-card {
            position: relative;
            overflow: hidden;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .image-card img {
            width: 100%;
            height: auto;
            display: block;
        }
        .image-caption {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(to top, rgba(0,0,0,0.7), transparent);
            color: white;
            padding: 8px;
            font-size: 12px;
            text-align: center;
        }
        .btn-loading {
            pointer-events: none;
            opacity: 0.7;
        }
        .toast-container {
            z-index: 9999;
        }
    `;
    document.head.appendChild(style);
});