        (function() {
            const canvas = document.getElementById('canvas-spheres');
            const scene = new THREE.Scene();

            // Set canvas dimensions to fit within text column
            const aspectRatio = 16 / 9;
            let canvasWidth = Math.min(800, canvas.parentElement.clientWidth || 800);
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
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 3));
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

            function drawCue() {
                ctx.save();
                ctx.setTransform(1, 0, 0, 1, 0, 0);
                ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
                ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

                const centerX = canvasWidth / 2;
                const arrowY = canvasHeight * 0.94;
                const textY = canvasHeight * 0.97;

                ctx.save();
                ctx.strokeStyle = '#888888';
                ctx.fillStyle = '#888888';
                ctx.lineWidth = 2;
                ctx.font = '14px system-ui, -apple-system, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';

                // Left arrow
                ctx.save();
                ctx.translate(centerX - 55, arrowY);
                ctx.beginPath();
                const leftStartX = -25;
                const leftStartY = -2;
                const leftEndX = -5;
                const leftEndY = 8;
                const leftControlX = -18;
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

                ctx.fillText('click & drag', centerX, textY);

                // Right arrow
                ctx.save();
                ctx.translate(centerX + 55, arrowY);
                ctx.beginPath();
                const rightStartX = 25;
                const rightStartY = -2;
                const rightEndX = 5;
                const rightEndY = 8;
                const rightControlX = 18;
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

                // // Annotations
                // ctx.save();
                // ctx.strokeStyle = '#888888';
                // ctx.fillStyle = '#888888';
                // ctx.lineWidth = 1.5;
                // ctx.font = '12px system-ui, -apple-system, sans-serif';

                // // "Outliers" annotation - pointing to left side
                // const outlierX = canvasWidth * 0.15;
                // const outlierY = canvasHeight * 0.25;
                // ctx.textAlign = 'left';
                // ctx.fillText('Outliers', outlierX + 15, outlierY);

                // ctx.beginPath();
                // ctx.moveTo(outlierX, outlierY);
                // ctx.lineTo(outlierX + 10, outlierY);
                // ctx.stroke();

                // ctx.beginPath();
                // ctx.arc(outlierX, outlierY, 3, 0, Math.PI * 2);
                // ctx.fill();

                // // "Noisy synthetic samples" annotation - pointing to overlap region
                // const noisyX = canvasWidth * 0.5;
                // const noisyY = canvasHeight * 0.15;
                // ctx.textAlign = 'center';
                // ctx.fillText('Noisy synthetic', noisyX, noisyY - 10);
                // ctx.fillText('samples', noisyX, noisyY + 5);

                // ctx.beginPath();
                // ctx.moveTo(noisyX, noisyY + 15);
                // ctx.lineTo(noisyX, noisyY + 25);
                // ctx.stroke();

                // ctx.beginPath();
                // ctx.arc(noisyX, noisyY + 28, 3, 0, Math.PI * 2);
                // ctx.fill();

                // // "Authentic synthetic samples" annotation - pointing to right side
                // const authX = canvasWidth * 0.85;
                // const authY = canvasHeight * 0.4;
                // ctx.textAlign = 'right';
                // ctx.fillText('Authentic synthetic', authX - 15, authY - 7);
                // ctx.fillText('samples', authX - 15, authY + 7);

                // ctx.beginPath();
                // ctx.moveTo(authX, authY);
                // ctx.lineTo(authX - 10, authY);
                // ctx.stroke();

                // ctx.beginPath();
                // ctx.arc(authX, authY, 3, 0, Math.PI * 2);
                // ctx.fill();

                ctx.restore();
            }


            canvas.addEventListener('mouseenter', () => {});

            canvas.addEventListener('mouseleave', () => {});

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
            directionalLight.shadow.mapSize.width = 2048;
            directionalLight.shadow.mapSize.height = 2048;
            directionalLight.shadow.normalBias = 0.02;
            scene.add(directionalLight);

            const pointLight = new THREE.PointLight(0xffffff, 0.5);
            pointLight.position.set(-5, 5, 5);
            scene.add(pointLight);


            const rotationGroup = new THREE.Group();
            scene.add(rotationGroup);

            // Create blue sphere (left) - FIXED: removed side: THREE.DoubleSide
            const radius = 4;
            const blueGeometry = new THREE.SphereGeometry(radius, 64, 64);
            const blueMaterial = new THREE.MeshPhongMaterial({
                color: 0x5dade2,
                transparent: true,
                opacity: 0.7,
                shininess: 30
            });
            const blueSphere = new THREE.Mesh(blueGeometry, blueMaterial);
            blueSphere.position.set(-2, 0, 0);
            rotationGroup.add(blueSphere);

            // Create red sphere (right) - FIXED: removed side: THREE.DoubleSide
            const redGeometry = new THREE.SphereGeometry(radius * 0.7, 64, 64);
            const redMaterial = new THREE.MeshPhongMaterial({
                color: 0xe74c3c,
                transparent: true,
                opacity: 0.7,
                shininess: 30
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

            // Mouse controls - FIXED: centered rotation
            let mouseDown = false;
            let lastMouseX = 0;
            let lastMouseY = 0;

            canvas.addEventListener('mousedown', (event) => {
                mouseDown = true;
                const rect = canvas.getBoundingClientRect();
                lastMouseX = event.clientX - rect.left;
                lastMouseY = event.clientY - rect.top;
            });

            canvas.addEventListener('mouseup', () => mouseDown = false);
            canvas.addEventListener('mouseleave', () => mouseDown = false);

            canvas.addEventListener('mousemove', (event) => {
                if (mouseDown) {
                    const rect = canvas.getBoundingClientRect();
                    const mouseX = event.clientX - rect.left;
                    const mouseY = event.clientY - rect.top;

                    const deltaX = (mouseX - lastMouseX) * 0.01;
                    const deltaY = (mouseY - lastMouseY) * 0.01;

                    rotationGroup.rotation.y += deltaX;
                    rotationGroup.rotation.x += deltaY;

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

                drawCue();
            });

            function animate() {
                requestAnimationFrame(animate);
                renderer.render(scene, camera);
            }

            drawCue();
            animate();
        })();
