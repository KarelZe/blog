(function() { // IIFE to avoid variable conflicts with other Three.js scripts
// Scene setup
const canvas = document.getElementById('canvas-tangent-space');
const scene = new THREE.Scene();

// Set canvas dimensions to fit within text column
const aspectRatio = 16 / 9;
let canvasWidth = Math.min(800, canvas.parentElement.clientWidth || 800); // Max width 800px
let canvasHeight = canvasWidth / aspectRatio;

const camera = new THREE.PerspectiveCamera(
    75,
    aspectRatio,
    0.1,
    1000
);
camera.position.set(8, 5, 8);

// Slightly lower the sphere on the canvas by lowering the group center's y coordinate
const sphereCenter = new THREE.Vector3(-1.5, 0.0, -0.9); // was y=1.5, now y=0.7

// Ensure parent container has a max-width
if (canvas.parentElement) {
    canvas.parentElement.style.maxWidth = '100%';
}

const renderer = new THREE.WebGLRenderer({
    canvas: canvas,  // Use the existing canvas element
    antialias: true,
    alpha: true
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 3)); // Cap pixel ratio
renderer.setSize(canvasWidth, canvasHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

// Create 2D overlay canvas for UI elements
const overlayCanvas = document.createElement('canvas');
overlayCanvas.width = canvasWidth * window.devicePixelRatio;
overlayCanvas.height = canvasHeight * window.devicePixelRatio;
overlayCanvas.style.position = 'absolute';
overlayCanvas.style.top = '0';
overlayCanvas.style.left = '0';
overlayCanvas.style.width = canvasWidth + 'px';
overlayCanvas.style.height = canvasHeight + 'px';
overlayCanvas.style.pointerEvents = 'none';

// Ensure parent has relative positioning
const canvasParent = canvas.parentElement;
canvasParent.style.position = 'relative';
canvasParent.appendChild(overlayCanvas);

const ctx = overlayCanvas.getContext('2d');
ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

// Hover state for fade effect
let cueOpacity = 0;
let targetOpacity = 0;
let hasInteracted = false;  // Track if user has clicked and dragged

// Draw click & drag cue
function drawCue() {
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset transform
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Skip drawing if fully transparent
    if (cueOpacity < 0.01) {
        ctx.restore();
        return;
    }

    const centerX = canvasWidth / 2;
    arrowY = canvasHeight * 0.94;
    textY = canvasHeight * 0.97;

    ctx.save();
    ctx.strokeStyle = '#888888';
    ctx.fillStyle = '#888888';
    ctx.lineWidth = 2;
    ctx.font = '14px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.globalAlpha = cueOpacity;

    // Left curved arrow
    ctx.save();
    ctx.translate(centerX - 55, arrowY);
    ctx.beginPath();

    // Use quadratic curve for smooth arc
    const leftStartX = -25;
    const leftStartY = -2;
    const leftEndX = -5;
    const leftEndY = 8;
    const leftControlX = -18;  // Control point for curve
    const leftControlY = 4;    // Reduced curve - midway between start and end Y

    ctx.moveTo(leftStartX, leftStartY);
    ctx.quadraticCurveTo(leftControlX, leftControlY, leftEndX, leftEndY);
    ctx.stroke();

    // Calculate tangent angle at the START of the curve
    // For quadratic curve, tangent at start point is from start point to control point
    const leftTangentAngle = Math.atan2(leftControlY - leftStartY, leftControlX - leftStartX);

    // Arrow head slightly before the curve start (backward along tangent)
    const arrowOffset = 4; // Distance to move arrow head backward
    const leftArrowX = leftStartX - Math.cos(leftTangentAngle) * arrowOffset;
    const leftArrowY = leftStartY - Math.sin(leftTangentAngle) * arrowOffset;

    ctx.save();
    ctx.translate(leftArrowX, leftArrowY);
    ctx.rotate(leftTangentAngle + Math.PI); // Add PI to point backwards
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(-8, -4);
    ctx.lineTo(-8, 4);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    ctx.restore();

    // Text (slightly more transparent than arrows)
    ctx.globalAlpha = cueOpacity * 0.8;
    ctx.fillText('click & drag', centerX, textY);
    ctx.globalAlpha = cueOpacity;

    // Right curved arrow
    ctx.save();
    ctx.translate(centerX + 55, arrowY);
    ctx.beginPath();

    // Use quadratic curve for smooth arc (mirrored from left)
    const rightStartX = 25;
    const rightStartY = -2;
    const rightEndX = 5;
    const rightEndY = 8;
    const rightControlX = 18;   // Control point for curve
    const rightControlY = 4;    // Reduced curve - midway between start and end Y

    ctx.moveTo(rightStartX, rightStartY);
    ctx.quadraticCurveTo(rightControlX, rightControlY, rightEndX, rightEndY);
    ctx.stroke();

    // Calculate tangent angle at the START of the curve
    // For quadratic curve, tangent at start point is from start point to control point
    const rightTangentAngle = Math.atan2(rightControlY - rightStartY, rightControlX - rightStartX);

    // Arrow head slightly before the curve start (backward along tangent)
    const rightArrowOffset = 4; // Distance to move arrow head backward
    const rightArrowX = rightStartX - Math.cos(rightTangentAngle) * rightArrowOffset;
    const rightArrowY = rightStartY - Math.sin(rightTangentAngle) * rightArrowOffset;

    ctx.save();
    ctx.translate(rightArrowX, rightArrowY);
    ctx.rotate(rightTangentAngle + Math.PI); // Add PI to point backwards along the curve
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(-8, -4);
    ctx.lineTo(-8, 4);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    ctx.restore();

    ctx.restore();
    ctx.restore(); // Match the initial ctx.save()
}

// Mouse hover events for cue fade in/out
canvas.addEventListener('mouseenter', () => {
    if (!hasInteracted) {
        targetOpacity = 1;
    }
});

canvas.addEventListener('mouseleave', () => {
    targetOpacity = 0;
});

// Lighting
const ambientLight = new THREE.AmbientLight(0x303030);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
directionalLight.position.set(5, 10, 5);
directionalLight.castShadow = true;
directionalLight.shadow.camera.near = 0.1;
directionalLight.shadow.camera.far = 50;
directionalLight.shadow.camera.left = -10;
directionalLight.shadow.camera.right = 10;
directionalLight.shadow.camera.top = 10;
directionalLight.shadow.camera.bottom = -10;
// Improve shadow quality
directionalLight.shadow.mapSize.width = 2048;
directionalLight.shadow.mapSize.height = 2048;
directionalLight.shadow.normalBias = 0.02;
scene.add(directionalLight);

const pointLight = new THREE.PointLight(0xffffff, 0.5);
pointLight.position.set(-5, 5, 5);
scene.add(pointLight);

// Create a group to hold all objects - positioned at sphere center for proper rotation
const rotationGroup = new THREE.Group();
rotationGroup.position.copy(sphereCenter);
scene.add(rotationGroup);

// Create sphere at origin of the group
const scale = 7.5;
const sphereGeometry = new THREE.SphereGeometry(scale, 64, 64);
const sphereMaterial = new THREE.MeshPhongMaterial({
    color: 0x4169e1,
    shininess: 10,  // Much lower shininess for matte appearance
    specular: 0x050505,  // Minimal specular reflection
    transparent: false,
    opacity: 1.0,
    side: THREE.DoubleSide
});
const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
sphere.position.set(0, 0, 0);  // Sphere at group origin
sphere.castShadow = true;
sphere.receiveShadow = true;
rotationGroup.add(sphere);

// Calculate tangent point on sphere (relative to group origin)
const point = new THREE.Vector3(
    scale / Math.sqrt(2.5),
    scale / Math.sqrt(2.5),
    scale / Math.sqrt(5)
);

// Create tangent plane
const planeSize = 5.75;  // Matched to sphere size
const planeGeometry = new THREE.PlaneGeometry(planeSize, planeSize, 10, 10);
const planeMaterial = new THREE.MeshPhongMaterial({
    color: 0xDC143C,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.8,
    shininess: 50
});
const tangentPlane = new THREE.Mesh(planeGeometry, planeMaterial);

// Position and orient the tangent plane
tangentPlane.position.copy(point);

// Calculate normal at the point (for sphere, normal = normalized position from center)
// Since sphere is now at origin of group, the normal is just the normalized point position
const normal = point.clone().normalize();

// Orient plane perpendicular to normal
tangentPlane.lookAt(point.clone().add(normal));
tangentPlane.castShadow = true;
tangentPlane.receiveShadow = true;
rotationGroup.add(tangentPlane);

// Add point marker
const pointGeometry = new THREE.SphereGeometry(0.19, 32, 32);  // Proportional to sphere size
const pointMaterial = new THREE.MeshPhongMaterial({
    color: 0x000000,
    emissive: 0x222222
});
const pointMarker = new THREE.Mesh(pointGeometry, pointMaterial);
pointMarker.position.copy(point);
rotationGroup.add(pointMarker);

// Add grid lines on tangent plane
const gridHelper = new THREE.GridHelper(planeSize, 10, 0x880000, 0x660000);
gridHelper.position.copy(point);
gridHelper.lookAt(point.clone().add(normal));
gridHelper.rotateX(Math.PI / 2);
rotationGroup.add(gridHelper);

// Set initial orientation
rotationGroup.rotation.x = -0.18;
rotationGroup.rotation.y = 0.27;
rotationGroup.rotation.z = 0;

// Mouse controls
let mouseX = 0;
let mouseY = 0;
let lastMouseX = 0;
let lastMouseY = 0;
let mouseDown = false;

canvas.addEventListener('mousedown', (event) => {
    mouseDown = true;
    const rect = canvas.getBoundingClientRect();
    lastMouseX = (event.clientX - rect.left - rect.width / 2) * 0.005; // Reduced from 0.01 to 0.005
    lastMouseY = (event.clientY - rect.top - rect.height / 2) * 0.005; // Slower rotation

    // Fade out cue on first interaction
    if (!hasInteracted) {
        hasInteracted = true;
        targetOpacity = 0;
    }
});
canvas.addEventListener('mouseup', () => mouseDown = false);
canvas.addEventListener('mouseleave', () => mouseDown = false); // Stop rotation when mouse leaves canvas

canvas.addEventListener('mousemove', (event) => {
    if (mouseDown) {
        const rect = canvas.getBoundingClientRect();
        mouseX = (event.clientX - rect.left - rect.width / 2) * 0.005; // Reduced from 0.01 to 0.005
        mouseY = (event.clientY - rect.top - rect.height / 2) * 0.005; // Slower rotation
        // Directly set rotation based on drag delta
        rotationGroup.rotation.y += (mouseX - lastMouseX);
        rotationGroup.rotation.x += (mouseY - lastMouseY);
        lastMouseX = mouseX;
        lastMouseY = mouseY;
    }
});

// Window resize
window.addEventListener('resize', () => {
    const newWidth = Math.min(800, canvas.parentElement.clientWidth || 800);
    const newHeight = newWidth / aspectRatio;

    camera.aspect = aspectRatio;
    camera.updateProjectionMatrix();
    renderer.setSize(newWidth, newHeight);

    // Resize overlay canvas
    overlayCanvas.width = newWidth * window.devicePixelRatio;
    overlayCanvas.height = newHeight * window.devicePixelRatio;
    overlayCanvas.style.width = newWidth + 'px';
    overlayCanvas.style.height = newHeight + 'px';

    // Reset context after resize
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Update canvas dimensions for drawCue
    canvasWidth = newWidth;
    canvasHeight = newHeight;

    if (cueOpacity > 0.01) drawCue();
});

// Animation loop
function animate() {
    requestAnimationFrame(animate);

    // Animate cue opacity
    if (Math.abs(cueOpacity - targetOpacity) > 0.01) {
        cueOpacity += (targetOpacity - cueOpacity) * 0.1; // Smooth transition
        drawCue();
    }

    // No auto-rotation, no smooth rotation: only update from drag
    camera.lookAt(sphereCenter); // Look at the sphere's center
    renderer.render(scene, camera);
}

animate();
})(); // End of IIFE
