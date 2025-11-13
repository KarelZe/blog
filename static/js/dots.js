(function() {
            const canvas = document.getElementById('canvas-spheres');
            const scene = new THREE.Scene();
            // scene.background = new THREE.Color(0xf5f5f5);

            // Set canvas dimensions to fit within text column
            const aspectRatio = 16 / 9;
            let canvasWidth = Math.min(800, canvas.parentElement.clientWidth || 800); // Max width 800px
            let canvasHeight = canvasWidth / aspectRatio;
            const camera = new THREE.PerspectiveCamera(75, aspectRatio, 0.1, 1000);
            camera.position.set(0, 0, 10);

            if (canvas.parentElement) {
                canvas.parentElement.style.maxWidth = '100%';
            }

            const renderer = new THREE.WebGLRenderer({
                canvas: canvas,
                antialias: true,
                alpha: true
            });
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 3)); // Cap pixel ratio
            renderer.setSize(canvasWidth, canvasHeight);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;

            // Create overlay canvas for UI
            const overlayCanvas = document.createElement('canvas');
            overlayCanvas.width = canvasWidth * window.devicePixelRatio;
            overlayCanvas.height = canvasHeight * window.devicePixelRatio;
            overlayCanvas.style.position = 'absolute';
            overlayCanvas.style.top = '0';
            overlayCanvas.style.left = '0';
            overlayCanvas.style.width = canvasWidth + 'px';
            overlayCanvas.style.height = canvasHeight + 'px';
            overlayCanvas.style.pointerEvents = 'none';

            const canvasParent = canvas.parentElement;
            canvasParent.style.position = 'relative';
            canvasParent.appendChild(overlayCanvas);

            const ctx = overlayCanvas.getContext('2d');
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

            let cueOpacity = 0;
            let targetOpacity = 0;
            let hasInteracted = false;

            function drawCue() {
                ctx.save();
                ctx.setTransform(1, 0, 0, 1, 0, 0);
                ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
                ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

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

                // Left arrow
                ctx.save();
                ctx.translate(centerX - 55, arrowY);
                ctx.beginPath();
                //const leftStartX = -25, leftStartY = -2;
                // const leftEndX = -5, leftEndY = 8;
                // const leftControlX = -18, leftControlY = 4;
                const leftStartX = -25;
                const leftStartY = -2;
                const leftEndX = -5;
                const leftEndY = 8;
                const leftControlX = -18;  // Control point for curve
                const leftControlY = 4;
                ctx.moveTo(leftStartX, leftStartY);
                ctx.quadraticCurveTo(leftControlX, leftControlY, leftEndX, leftEndY);
                ctx.stroke();

                const leftTangentAngle = Math.atan2(leftControlY - leftStartY, leftControlX - leftStartX);
                const arrowOffset = 4;
                const leftArrowX = leftStartX - Math.cos(leftTangentAngle) * arrowOffset;
                const leftArrowY = leftStartY - Math.sin(leftTangentAngle) * arrowOffset;

                ctx.save();
                ctx.translate(leftArrowX, leftArrowY);
                ctx.rotate(leftTangentAngle + Math.PI);
                ctx.beginPath();
                ctx.moveTo(0, 0);
                ctx.lineTo(-8, -4);
                ctx.lineTo(-8, 4);
                ctx.closePath();
                ctx.fill();
                ctx.restore();
                ctx.restore();

                ctx.globalAlpha = cueOpacity * 0.8;
                ctx.fillText('click & drag', centerX, textY);
                ctx.globalAlpha = cueOpacity;

                // Right arrow
                ctx.save();
                ctx.translate(centerX + 55, arrowY);
                ctx.beginPath();
                // const rightStartX = 25, rightStartY = -2;
                // const rightEndX = 5, rightEndY = 8;
                // const rightControlX = 18, rightControlY = 4;

                const rightStartX = 25;
                const rightStartY = -2;
                const rightEndX = 5;
                const rightEndY = 8;
                const rightControlX = 18;   // Control point for curve
                const rightControlY = 4;

                ctx.moveTo(rightStartX, rightStartY);
                ctx.quadraticCurveTo(rightControlX, rightControlY, rightEndX, rightEndY);
                ctx.stroke();

                const rightTangentAngle = Math.atan2(rightControlY - rightStartY, rightControlX - rightStartX);
                const rightArrowX = rightStartX - Math.cos(rightTangentAngle) * arrowOffset;
                const rightArrowY = rightStartY - Math.sin(rightTangentAngle) * arrowOffset;

                ctx.save();
                ctx.translate(rightArrowX, rightArrowY);
                ctx.rotate(rightTangentAngle + Math.PI);
                ctx.beginPath();
                ctx.moveTo(0, 0);
                ctx.lineTo(-8, -4);
                ctx.lineTo(-8, 4);
                ctx.closePath();
                ctx.fill();
                ctx.restore();
                ctx.restore();

                ctx.restore();
            }


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
            directionalLight.position.set(5, 5, 5);
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


            const rotationGroup = new THREE.Group();
            scene.add(rotationGroup);

            // Create blue sphere (left)
            const radius = 4;
            const blueGeometry = new THREE.SphereGeometry(radius, 64, 64);
            const blueMaterial = new THREE.MeshPhongMaterial({
                color: 0x5dade2,
                transparent: true,
                opacity: 0.7,
                shininess: 30,
                side: THREE.DoubleSide
            });
            const blueSphere = new THREE.Mesh(blueGeometry, blueMaterial);
            blueSphere.position.set(-2, 0, 0);
            rotationGroup.add(blueSphere);

            // Create red sphere (right)
            const redGeometry = new THREE.SphereGeometry(radius, 64, 64);
            const redMaterial = new THREE.MeshPhongMaterial({
                color: 0xe74c3c,
                transparent: true,
                opacity: 0.7,
                shininess: 30,
                side: THREE.DoubleSide
            });
            const redSphere = new THREE.Mesh(redGeometry, redMaterial);
            redSphere.position.set(2, 0, 1);
            rotationGroup.add(redSphere);

            // Create dots
            const dotGeometry = new THREE.SphereGeometry(0.15, 16, 16);

            // Blue dots inside blue sphere
            for (let i = 0; i < 15; i++) {
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.random() * Math.PI;
                const r = Math.random() * radius;

                const x = r * Math.sin(phi) * Math.cos(theta) - 2;
                const y = r * Math.sin(phi) * Math.sin(theta);
                const z = r * Math.cos(phi);

                const dotMaterial = new THREE.MeshPhongMaterial({
                    color: 0x2874a6,
                    shininess: 50
                });
                const dot = new THREE.Mesh(dotGeometry, dotMaterial);
                dot.position.set(x, y, z);
                rotationGroup.add(dot);
            }

            // Red dots inside red sphere
            for (let i = 0; i < 15; i++) {
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.random() * Math.PI;
                const r = Math.random() * 0.4 * radius;

                const x = r * Math.sin(phi) * Math.cos(theta) + 2;
                const y = r * Math.sin(phi) * Math.sin(theta);
                const z = r * Math.cos(phi);

                const dotMaterial = new THREE.MeshPhongMaterial({
                    color: 0xa93226,
                    shininess: 50
                });
                const dot = new THREE.Mesh(dotGeometry, dotMaterial);
                dot.position.set(x, y, z);
                rotationGroup.add(dot);
            }

            // Blue outliers
            for (let i = 0; i < 8; i++) {
                const angle = Math.random() * Math.PI * 2;
                const distance = radius + 0.5 + Math.random() * 2;
                const height = (Math.random() - 0.5) * 8;

                const x = Math.cos(angle) * distance - 2;
                const y = height;
                const z = Math.sin(angle) * distance;

                const dotMaterial = new THREE.MeshPhongMaterial({
                    color: 0x2874a6,
                    shininess: 50
                });
                const dot = new THREE.Mesh(dotGeometry, dotMaterial);
                dot.position.set(x, y, z);
                rotationGroup.add(dot);
            }

            // Red outliers
            for (let i = 0; i < 8; i++) {
                const angle = Math.random() * Math.PI * 2;
                const distance = radius + 0.5 + Math.random() * 2;
                const height = (Math.random() - 0.5) * 8;

                const x = Math.cos(angle) * distance + 2;
                const y = height;
                const z = Math.sin(angle) * distance;

                const dotMaterial = new THREE.MeshPhongMaterial({
                    color: 0xa93226,
                    shininess: 50
                });
                const dot = new THREE.Mesh(dotGeometry, dotMaterial);
                dot.position.set(x, y, z);
                rotationGroup.add(dot);
            }

            // Add coordinate axes
            // const axesHelper = new THREE.AxesHelper(8);
            // rotationGroup.add(axesHelper);

            // Mouse controls
            let mouseDown = false;
            let lastMouseX = 0;
            let lastMouseY = 0;

            canvas.addEventListener('mousedown', (event) => {
                mouseDown = true;
                const rect = canvas.getBoundingClientRect();
                lastMouseX = (event.clientX - rect.left - rect.width / 2) * 0.005;
                lastMouseY = (event.clientY - rect.top - rect.height / 2) * 0.005;

                if (!hasInteracted) {
                    hasInteracted = true;
                    targetOpacity = 0;
                }
            });

            canvas.addEventListener('mouseup', () => mouseDown = false);
            canvas.addEventListener('mouseleave', () => mouseDown = false);

            canvas.addEventListener('mousemove', (event) => {
                if (mouseDown) {
                    const rect = canvas.getBoundingClientRect();
                    const mouseX = (event.clientX - rect.left - rect.width / 2) * 0.005;
                    const mouseY = (event.clientY - rect.top - rect.height / 2) * 0.005;

                    rotationGroup.rotation.y += (mouseX - lastMouseX);
                    rotationGroup.rotation.x += (mouseY - lastMouseY);

                    lastMouseX = mouseX;
                    lastMouseY = mouseY;
                }
            });

            window.addEventListener('resize', () => {
            const newWidth = Math.min(800, canvas.parentElement.clientWidth || 800);
            const newHeight = newWidth / aspectRatio;

            camera.aspect = aspectRatio;
                camera.updateProjectionMatrix();
                renderer.setSize(newWidth, newHeight);

                overlayCanvas.width = newWidth * window.devicePixelRatio;
                overlayCanvas.height = newHeight * window.devicePixelRatio;
                overlayCanvas.style.width = newWidth + 'px';
                overlayCanvas.style.height = newHeight + 'px';

                ctx.setTransform(1, 0, 0, 1, 0, 0);
                ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

                canvasWidth = newWidth;
                canvasHeight = newHeight;

                if (cueOpacity > 0.01) drawCue();
            });

            // Set initial rotation
            // rotationGroup.rotation.x = -0.2;
            // rotationGroup.rotation.y = 0.3;

            function animate() {
                requestAnimationFrame(animate);

                if (Math.abs(cueOpacity - targetOpacity) > 0.01) {
                    cueOpacity += (targetOpacity - cueOpacity) * 0.1;
                    drawCue();
                }

                renderer.render(scene, camera);
            }

            animate();
        })();
