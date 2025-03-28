// Initialize pan-related variables
let isPanning = false;
let lastPointerPosition;
let isPanKeyActive = false;

// Get user hash or prompt for name
let userHash = localStorage.getItem('userHash');
if (!userHash || userHash === 'undefined' || userHash === 'null' || userHash.trim() === '' || userHash === null) {
    localStorage.removeItem('userHash');
    let userName = null;
    do {
        userName = prompt("Please enter your name or other moniker - this is not stored, it is hashed, anonymising you, and used for file management - differenciating different annotaitons of the same file:");
        if (userName !== null && userName.trim() !== '') {
            userHash = btoa(userName.trim()).substring(0, 6);
            localStorage.setItem('userHash', userHash);
        } else if (userName === null) {
            alert("You must enter a name to proceed.");
        } else {
            alert("Name cannot be empty.");
        }
    } while (!userHash);
}

// Initialize variables
let currentIndex = parseInt(localStorage.getItem('currentIndex')) || 0;
let images = [];
let stage, layer, imageLayer;
let drawingTool = 'pan';
let isDrawing = false;
let tempShape = null;
let currentLine = null;
let history = [];
let historyStep = -1;

// Add variables for image boundaries
let imageBounds = {
    x: 0,
    y: 0,
    width: 0,
    height: 0
};
let currentImage = null;

// Calculate initial stage dimensions
function getStageSize() {
    const headerHeight = document.querySelector('.header').offsetHeight;
    const footerHeight = document.querySelector('#footer').offsetHeight;
    const windowHeight = window.innerHeight;
    const availableHeight = windowHeight - headerHeight;

    return {
        width: window.innerWidth,
        height: availableHeight
    };
}

// Calculate initial scale for 2000x2000 images
function calculateInitialScale(stageWidth, stageHeight) {
    const padding = 50; // 25px padding on each side
    const availableWidth = stageWidth - padding;
    const availableHeight = stageHeight - padding;

    // For 2000x2000 images
    const imageSize = 2000;
    return Math.min(availableWidth / imageSize, availableHeight / imageSize);
}

// Handle window resize
function handleResize() {
    if (!stage) return;

    const size = getStageSize();
    stage.width(size.width);
    stage.height(size.height);
    stage.batchDraw();
}

// Add resize listener
window.addEventListener('resize', handleResize);

// Fetch images and start
fetch('/images')
    .then(response => response.json())
    .then(data => {
        images = data;
        updateProgress();
        loadImage();
    });

function updateProgress() {
    let totalImages = images.length;
    let progressBar = document.getElementById('progress-bar');
    let progressText = document.getElementById('progress-text');
    progressBar.max = totalImages;
    progressBar.value = currentIndex;
    progressText.textContent = `${currentIndex} / ${totalImages} images annotated`;
}

function loadImage() {
    if (currentIndex >= images.length) {
        alert('All images have been annotated. Thank you!');
        return;
    }
    currentImage = images[currentIndex];
    let imageObj = new Image();
    imageObj.onload = function() {
        setupStage(imageObj);
    };
    imageObj.src = '/static/images/' + currentImage;
}

// Constrain point to image boundaries
function constrainToImage(pos) {
    // Convert stage coordinates to image coordinates
    const transform = stage.getAbsoluteTransform().copy();
    transform.invert();
    const stagePos = transform.point(pos);

    // Constrain to image boundaries
    return {
        x: Math.max(imageBounds.x, Math.min(stagePos.x, imageBounds.x + imageBounds.width)),
        y: Math.max(imageBounds.y, Math.min(stagePos.y, imageBounds.y + imageBounds.height))
    };
}

function updateCursor() {
    const container = document.getElementById('container');
    container.classList.remove(
        'pan-cursor',
        'rectangle-cursor',
        'circle-cursor',
        'freehand-cursor',
        'zoom-in-cursor',
        'zoom-out-cursor'
    );

    if (isPanKeyActive || drawingTool === 'pan') {
        container.classList.add('pan-cursor');
    } else {
        container.classList.add(`${drawingTool}-cursor`);
    }
}

function setupStage(imageObj) {
    if (stage) stage.destroy();

    const stageSize = getStageSize();

    // Create stage with full window size and willReadFrequently attribute
    stage = new Konva.Stage({
        container: 'container',
        width: stageSize.width,
        height: stageSize.height,
        createJSCanvas: true,
        pixelRatio: 1,
        canvas: (function() {
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d', {
                willReadFrequently: true
            });
            return canvas;
        })()
    });

    // Set up layers with optimized canvas contexts
    imageLayer = new Konva.Layer({
        canvas: (function() {
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d', {
                willReadFrequently: true
            });
            return canvas;
        })()
    });

    layer = new Konva.Layer({
        canvas: (function() {
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d', {
                willReadFrequently: true
            });
            return canvas;
        })()
    });

    stage.add(imageLayer);
    stage.add(layer);

    // Add image to imageLayer
    let bgImage = new Konva.Image({
        x: 0,
        y: 0,
        image: imageObj,
        width: imageObj.width, // Use actual image dimensions
        height: imageObj.height
    });
    imageLayer.add(bgImage);

    // Update image boundaries based on the actual image
    imageBounds = {
        x: 0,
        y: 0,
        width: imageObj.width,
        height: imageObj.height
    };

    // Calculate and apply initial scale and position
    const initialScale = calculateInitialScale(stageSize.width, stageSize.height);
    const centerX = (stageSize.width - (imageObj.width * initialScale)) / 2;
    const centerY = (stageSize.height - (imageObj.height * initialScale)) / 2;

    stage.scale({ x: initialScale, y: initialScale });
    stage.position({ x: centerX, y: centerY });

    imageLayer.draw();

    // Initialize history
    history = [];
    historyStep = -1;
    addHistory();

    // Event listeners for drawing tools
    stage.on('mousedown touchstart', function(e) {
        const isPanMode = drawingTool === 'pan' || isPanKeyActive;

        if (isPanMode) {
            isPanning = true;
            lastPointerPosition = stage.getPointerPosition();
            stage.container().style.cursor = 'grabbing';
            return;
        }

        isDrawing = true;
        // Get pointer position and constrain to image boundaries
        let pos = constrainToImage(stage.getPointerPosition());

        if (drawingTool === 'rectangle') {
            tempShape = new Konva.Rect({
                x: pos.x,
                y: pos.y,
                width: 0,
                height: 0,
                fill: 'white',
                strokeWidth: 0
            });
            layer.add(tempShape);
        } else if (drawingTool === 'circle') {
            tempShape = new Konva.Ellipse({
                x: pos.x,
                y: pos.y,
                radiusX: 0,
                radiusY: 0,
                fill: 'white',
                strokeWidth: 0
            });
            layer.add(tempShape);
        } else if (drawingTool === 'freehand') {
            currentLine = new Konva.Line({
                stroke: 'white',
                strokeWidth: 2,
                points: [pos.x, pos.y],
                lineCap: 'round',
                lineJoin: 'round',
                closed: false,
                fill: 'white'
            });
            layer.add(currentLine);
        }
    });

    stage.on('mousemove touchmove', function(e) {
        if (isPanning && (drawingTool === 'pan' || isPanKeyActive)) {
            e.evt.preventDefault();
            const pos = stage.getPointerPosition();
            const dx = pos.x - lastPointerPosition.x;
            const dy = pos.y - lastPointerPosition.y;

            stage.position({
                x: stage.x() + dx,
                y: stage.y() + dy
            });
            stage.batchDraw();
            lastPointerPosition = pos;
            return;
        }

        if (!isDrawing) return;

        // Get constrained pointer position
        let pos = constrainToImage(stage.getPointerPosition());

        if (drawingTool === 'rectangle') {
            let newWidth = pos.x - tempShape.x();
            let newHeight = pos.y - tempShape.y();

            // Ensure rectangle stays within image bounds if drawn backwards
            if (newWidth < 0) {
                const newX = Math.max(imageBounds.x, tempShape.x() + newWidth);
                newWidth = tempShape.x() - newX;
                tempShape.x(newX);
            }

            if (newHeight < 0) {
                const newY = Math.max(imageBounds.y, tempShape.y() + newHeight);
                newHeight = tempShape.y() - newY;
                tempShape.y(newY);
            }

            tempShape.width(newWidth);
            tempShape.height(newHeight);
            layer.batchDraw();
        } else if (drawingTool === 'circle') {
            // Calculate radii based on constrained position
            let radiusX = Math.abs(pos.x - tempShape.x());
            let radiusY = Math.abs(pos.y - tempShape.y());

            // Additional constraint to ensure ellipse doesn't exceed image boundaries
            const centerX = tempShape.x();
            const centerY = tempShape.y();

            // Constrain radiusX to prevent exceeding left/right boundaries
            radiusX = Math.min(radiusX,
                Math.min(centerX - imageBounds.x, imageBounds.x + imageBounds.width - centerX));

            // Constrain radiusY to prevent exceeding top/bottom boundaries
            radiusY = Math.min(radiusY,
                Math.min(centerY - imageBounds.y, imageBounds.y + imageBounds.height - centerY));

            tempShape.radiusX(radiusX);
            tempShape.radiusY(radiusY);
            layer.batchDraw();
        } else if (drawingTool === 'freehand') {
            let newPoints = currentLine.points().concat([pos.x, pos.y]);
            currentLine.points(newPoints);
            layer.batchDraw();
        }
    });

    stage.on('mouseup touchend', function(e) {
        if (isPanning) {
            isPanning = false;
            stage.container().style.cursor = 'grab';
            return;
        }

        isDrawing = false;
        if (drawingTool === 'rectangle' || drawingTool === 'circle') {
            // Check if shape is too small
            if ((drawingTool === 'rectangle' &&
                (tempShape.width() < 5 || tempShape.height() < 5)) ||
                (drawingTool === 'circle' &&
                (tempShape.radiusX() < 3 || tempShape.radiusY() < 3))) {
                tempShape.destroy();
                layer.batchDraw();
            } else {
                addHistory();
            }
            tempShape = null;
        } else if (drawingTool === 'freehand') {
            // Check if line has enough points
            if (currentLine.points().length <= 4) {
                currentLine.destroy();
                layer.batchDraw();
            } else {
                currentLine.closed(true);
                layer.batchDraw();
                addHistory();
            }
            currentLine = null;
        }
    });

    // Zooming with mouse wheel
    stage.on('wheel', function(e) {
        e.evt.preventDefault();
        var oldScale = stage.scaleX();
        var pointer = stage.getPointerPosition();

        var scaleBy = 1.05;
        var direction = e.evt.deltaY > 0 ? -1 : 1;
        var newScale = direction > 0 ? oldScale * scaleBy : oldScale / scaleBy;

        // Allow more zoom out for full view
        newScale = Math.max(0.1, Math.min(newScale, 10));

        var mousePointTo = {
            x: (pointer.x - stage.x()) / oldScale,
            y: (pointer.y - stage.y()) / oldScale,
        };

        stage.scale({ x: newScale, y: newScale });

        var newPos = {
            x: pointer.x - mousePointTo.x * newScale,
            y: pointer.y - mousePointTo.y * newScale,
        };
        stage.position(newPos);
        stage.batchDraw();
    });

    stage.on('mouseover', updateCursor);
}

// Tool button event listeners
function selectTool(tool) {
    drawingTool = tool;
    const toolButtons = document.querySelectorAll('.tool-button');
    toolButtons.forEach(button => {
        button.classList.remove('selected');
    });
    document.getElementById(tool).classList.add('selected');
    updateCursor();
}

// Add event listeners for all tools
document.getElementById('pan').addEventListener('click', () => {
    selectTool('pan');
});
document.getElementById('rectangle').addEventListener('click', () => {
    selectTool('rectangle');
});
document.getElementById('circle').addEventListener('click', () => {
    selectTool('circle');
});
document.getElementById('freehand').addEventListener('click', () => {
    selectTool('freehand');
});
document.getElementById('undo').addEventListener('click', () => {
    undo();
});
document.getElementById('redo').addEventListener('click', () => {
    redo();
});
document.getElementById('zoom-in').addEventListener('click', () => {
    zoomStage(1.25);
});
document.getElementById('zoom-out').addEventListener('click', () => {
    zoomStage(0.8);
});
document.getElementById('done').addEventListener('click', saveAnnotation);

// Add space bar event listeners for temporary pan tool
document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !isPanKeyActive) {
        isPanKeyActive = true;
        document.getElementById('container').classList.add('pan-cursor');
    }
});

document.addEventListener('keyup', (e) => {
    if (e.code === 'Space') {
        isPanKeyActive = false;
        updateCursor();
    }
});

function zoomStage(scaleBy) {
    var oldScale = stage.scaleX();
    var pointer = {
        x: stage.width() / 2,
        y: stage.height() / 2
    };

    var newScale = oldScale * scaleBy;
    // Allow more zoom out for full view
    newScale = Math.max(0.1, Math.min(newScale, 10));

    var mousePointTo = {
        x: (pointer.x - stage.x()) / oldScale,
        y: (pointer.y - stage.y()) / oldScale,
    };

    stage.scale({ x: newScale, y: newScale });

    var newPos = {
        x: pointer.x - mousePointTo.x * newScale,
        y: pointer.y - mousePointTo.y * newScale,
    };
    stage.position(newPos);
    stage.batchDraw();
}

function saveAnnotation() {
    console.log('Saving annotation...');
    if (!userHash) {
        alert('User hash is not set. Please reload the page.');
        return;
    }
    if (!currentImage) {
        alert('No image to save.');
        return;
    }

    // Create a temporary canvas with the exact dimensions of the original image
    const canvas = document.createElement('canvas');
    canvas.width = imageBounds.width;
    canvas.height = imageBounds.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    // Fill with black background (transparent areas)
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw all shapes in white (masked areas)
    ctx.fillStyle = 'white';
    ctx.strokeStyle = 'white';

    // Get all shapes from the layer
    let shapes = layer.getChildren();

    // Create a temporary Konva stage to properly render the shapes
    const tempStage = new Konva.Stage({
        container: document.createElement('div'),
        width: canvas.width,
        height: canvas.height
    });

    const tempLayer = new Konva.Layer();
    tempStage.add(tempLayer);

    // Add each shape to the temporary layer
    shapes.forEach(shape => {
        const clonedShape = shape.clone({
            fill: 'white',
            stroke: 'white'
        });
        tempLayer.add(clonedShape);
    });

    tempLayer.draw();

    // Get the data URL from the temporary layer
    let dataURL = tempLayer.toDataURL({
        mimeType: 'image/png',
        quality: 1,
        x: 0,
        y: 0,
        width: canvas.width,
        height: canvas.height
    });

    // Clean up
    tempStage.destroy();

    fetch('/save_annotation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image_name: currentImage,
            user_hash: userHash,
            mask: dataURL
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Response from server:', data);
        if (data.status === 'success') {
            currentIndex++;
            localStorage.setItem('currentIndex', currentIndex);
            updateProgress();
            loadImage();
        } else {
            alert('Failed to save annotation: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while saving the annotation.');
    });
}

// Undo/Redo functionality
function addHistory() {
    historyStep++;
    if (historyStep < history.length) {
        history = history.slice(0, historyStep);
    }
    history.push(layer.toJSON());
}

function undo() {
    if (historyStep > 0) {
        historyStep--;
        layer.destroyChildren();
        let oldState = Konva.Node.create(history[historyStep], 'container');
        layer.add(...oldState.getChildren());
        layer.draw();
    }
}

function redo() {
    if (historyStep < history.length - 1) {
        historyStep++;
        layer.destroyChildren();
        let oldState = Konva.Node.create(history[historyStep], 'container');
        layer.add(...oldState.getChildren());
        layer.draw();
    }
}
