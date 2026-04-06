// operation.js - Complete frontend functionality

const API_BASE = 'http://localhost:8080';

// State management
let uploadedImages = [];
let processingResults = {
    harris: null,
    sift: null,
    ssd: null,
    ncc: null
};

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imageList = document.getElementById('imageList');
const processBtn = document.getElementById('processBtn');

// Tab handling
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabId = btn.dataset.tab;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
        document.getElementById(`tab-${tabId}`).classList.add('active');
    });
});

// Upload handling
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--accent-color)';
});
uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'rgba(239, 102, 3, 0.3)';
});
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'rgba(239, 102, 3, 0.3)';
    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
    handleFiles(files);
});

fileInput.addEventListener('change', (e) => {
    handleFiles(Array.from(e.target.files));
});

async function handleFiles(files) {
    const base64Images = [];
    for (const file of files) {
        const base64 = await fileToBase64(file);
        base64Images.push(base64.split(',')[1]); // Remove data:image/...;base64,
    }
    
    // Upload to server
    try {
        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ images: base64Images })
        });
        const data = await response.json();
        if (data.success) {
            uploadedImages = files;
            updateImageList();
            processBtn.disabled = false;
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Failed to upload images');
    }
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function updateImageList() {
    if (uploadedImages.length === 0) {
        imageList.innerHTML = '<div class="empty-state">No images uploaded</div>';
        return;
    }
    
    imageList.innerHTML = '';
    uploadedImages.slice(0, 2).forEach((file, idx) => {
        const div = document.createElement('div');
        div.className = 'image-item';
        div.innerHTML = `
            <img src="${URL.createObjectURL(file)}" class="image-thumb" alt="Thumb">
            <div class="image-info">
                <p class="image-name">${file.name}</p>
                <p class="image-size">${(file.size / 1024).toFixed(0)} KB</p>
            </div>
            <button class="remove-image" data-index="${idx}"><i class="bi bi-x"></i></button>
        `;
        imageList.appendChild(div);
    });
    
    document.querySelectorAll('.remove-image').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const idx = parseInt(btn.dataset.index);
            uploadedImages.splice(idx, 1);
            updateImageList();
            if (uploadedImages.length < 2) processBtn.disabled = true;
        });
    });
}

// Process All
processBtn.addEventListener('click', async () => {
    if (uploadedImages.length < 2) {
        alert('Please upload at least 2 images');
        return;
    }
    
    processBtn.disabled = true;
    processBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Processing...';
    
    try {
        // Step 1: Harris Corner Detection
        await processHarris();
        
        // Step 2: SIFT Feature Extraction
        await processSIFT();
        
        // Step 3: SSD Matching
        await processSSD();
        
        // Step 4: NCC Matching
        await processNCC();
        
        // Update comparison tab
        updateComparisonTab();
        
    } catch (error) {
        console.error('Processing error:', error);
        alert('Error processing images: ' + error.message);
    } finally {
        processBtn.disabled = false;
        processBtn.innerHTML = '<i class="bi bi-play-fill"></i> Process All';
    }
});

async function processHarris() {
    showLoading('harris1Loading', true);
    showLoading('harris2Loading', true);
    
    try {
        const response = await fetch(`${API_BASE}/api/harris`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success && data.results) {
            // Update Image 1
            if (data.results[0]) {
                document.getElementById('harris1Img').src = `data:image/png;base64,${data.results[0].image}`;
                document.getElementById('harris1Count').innerText = `${data.results[0].num_corners} points`;
                document.getElementById('harris1Time').innerText = `${data.results[0].time_ms.toFixed(1)} ms`;
                document.getElementById('metricHarris').innerText = `${data.results[0].time_ms.toFixed(1)} ms`;
            }
            // Update Image 2
            if (data.results[1]) {
                document.getElementById('harris2Img').src = `data:image/png;base64,${data.results[1].image}`;
                document.getElementById('harris2Count').innerText = `${data.results[1].num_corners} points`;
                document.getElementById('harris2Time').innerText = `${data.results[1].time_ms.toFixed(1)} ms`;
            }
            processingResults.harris = data.results;
        }
    } catch (error) {
        console.error('Harris error:', error);
    } finally {
        showLoading('harris1Loading', false);
        showLoading('harris2Loading', false);
    }
}

async function processSIFT() {
    showLoading('sift1Loading', true);
    showLoading('sift2Loading', true);
    
    try {
        const response = await fetch(`${API_BASE}/api/sift`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success && data.results) {
            if (data.results[0]) {
                document.getElementById('sift1Img').src = `data:image/png;base64,${data.results[0].image}`;
                document.getElementById('sift1Count').innerText = `${data.results[0].num_keypoints} points`;
                document.getElementById('sift1Time').innerText = `${data.results[0].time_ms.toFixed(1)} ms`;
                document.getElementById('metricSIFT').innerText = `${data.results[0].time_ms.toFixed(1)} ms`;
            }
            if (data.results[1]) {
                document.getElementById('sift2Img').src = `data:image/png;base64,${data.results[1].image}`;
                document.getElementById('sift2Count').innerText = `${data.results[1].num_keypoints} points`;
                document.getElementById('sift2Time').innerText = `${data.results[1].time_ms.toFixed(1)} ms`;
            }
            processingResults.sift = data.results;
        }
    } catch (error) {
        console.error('SIFT error:', error);
    } finally {
        showLoading('sift1Loading', false);
        showLoading('sift2Loading', false);
    }
}

async function processSSD() {
    showLoading('ssdLoading', true);
    
    try {
        const response = await fetch(`${API_BASE}/api/match-ssd`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('ssdImg').src = `data:image/png;base64,${data.image}`;
            document.getElementById('ssdCount').innerText = `${data.num_matches} matches`;
            document.getElementById('ssdTime').innerText = `${data.time_ms.toFixed(1)} ms`;
            document.getElementById('metricSSD').innerText = `${data.time_ms.toFixed(1)} ms`;
            processingResults.ssd = data;
        }
    } catch (error) {
        console.error('SSD error:', error);
    } finally {
        showLoading('ssdLoading', false);
    }
}

async function processNCC() {
    showLoading('nccLoading', true);
    
    try {
        const response = await fetch(`${API_BASE}/api/match-ncc`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('nccImg').src = `data:image/png;base64,${data.image}`;
            document.getElementById('nccCount').innerText = `${data.num_matches} matches`;
            document.getElementById('nccTime').innerText = `${data.time_ms.toFixed(1)} ms`;
            document.getElementById('metricNCC').innerText = `${data.time_ms.toFixed(1)} ms`;
            processingResults.ncc = data;
        }
    } catch (error) {
        console.error('NCC error:', error);
    } finally {
        showLoading('nccLoading', false);
    }
}

function updateComparisonTab() {
    if (processingResults.harris) {
        if (processingResults.harris[0]) {
            document.getElementById('compareHarris1').src = document.getElementById('harris1Img').src;
        }
        if (processingResults.harris[1]) {
            document.getElementById('compareHarris2').src = document.getElementById('harris2Img').src;
        }
    }
    if (processingResults.sift) {
        if (processingResults.sift[0]) {
            document.getElementById('compareSift1').src = document.getElementById('sift1Img').src;
        }
        if (processingResults.sift[1]) {
            document.getElementById('compareSift2').src = document.getElementById('sift2Img').src;
        }
    }
}

function showLoading(elementId, show) {
    const element = document.getElementById(elementId);
    if (element) {
        if (show) element.classList.add('active');
        else element.classList.remove('active');
    }
}

// Initial state
processBtn.disabled = true;
