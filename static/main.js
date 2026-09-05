let scene, camera, renderer, controls, currentMesh;

function init3D() {
    const container = document.getElementById('canvas-container');
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x20221f);

    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(0, 0, 3);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.outputEncoding = THREE.sRGBEncoding;
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 1.2);
    scene.add(hemiLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight1.position.set(2, 4, 3);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
    dirLight2.position.set(-2, -2, -3);
    scene.add(dirLight2);

    window.addEventListener('resize', () => {
        if (!container) return;
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });

    animate();
}

function animate() {
    requestAnimationFrame(animate);
    if (controls) controls.update();
    if (renderer && scene && camera) renderer.render(scene, camera);
}

document.addEventListener('DOMContentLoaded', () => {
    init3D();

    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('image-input');
    const imgPreview = document.getElementById('image-preview');
    const dropzoneText = document.getElementById('dropzone-text');
    const depthScale = document.getElementById('depth_scale');
    const resolution = document.getElementById('resolution');

    function showPreview(file) {
        if (!file || !file.type.startsWith('image/')) return;
        imgPreview.src = URL.createObjectURL(file);
        imgPreview.classList.remove('hidden');
        if (dropzoneText) dropzoneText.classList.add('hidden');
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files[0]) showPreview(e.target.files[0]);
        });
    }

    if (dropzone) {
        ['dragenter', 'dragover'].forEach((eventName) => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach((eventName) => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            const file = e.dataTransfer.files[0];
            if (file && fileInput) {
                fileInput.files = e.dataTransfer.files;
                showPreview(file);
            }
        });
    }

    if (depthScale) {
        depthScale.addEventListener('input', (e) => {
            const valSpan = document.getElementById('scale-val');
            if (valSpan) valSpan.textContent = Number(e.target.value).toFixed(2);
        });
    }

    if (resolution) {
        resolution.addEventListener('input', (e) => {
            const valSpan = document.getElementById('res-val');
            if (valSpan) valSpan.textContent = e.target.value;
        });
    }

    const convertForm = document.getElementById('convert-form');
    if (convertForm) {
        convertForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!fileInput || !fileInput.files[0]) return;

            const loader = document.getElementById('loader');
            const badge = document.getElementById('status-badge');
            const detectionBar = document.getElementById('detection-bar');

            if (loader) loader.classList.remove('hidden');
            if (badge) badge.textContent = 'Generating 3D model...';
            if (detectionBar) detectionBar.classList.add('hidden');

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            formData.append('depth_scale', depthScale ? depthScale.value : '0.25');
            formData.append('resolution', resolution ? resolution.value : '256');
            formData.append('isolate_subject', document.getElementById('isolate_subject').checked);

            try {
                const res = await fetch('/api/convert', { method: 'POST', body: formData });
                const data = await res.json();

                if (!res.ok || data.status === 'error') {
                    const msg = data.message || 'Generation failed. Please try again.';
                    if (loader) loader.classList.add('hidden');
                    if (badge) badge.textContent = 'Error';
                    alert(msg);
                    return;
                }

                if (loader) loader.classList.add('hidden');
                if (badge) badge.textContent = 'Rendered';

                const CATEGORY_META = {
                    ball:      { icon: '🏀', label: 'Ball / Sports Sphere', pill: '3D Sphere',      pillClass: 'pill-sphere' },
                    door:      { icon: '🚪', label: 'Door / Window / Panel', pill: '3D Mesh',        pillClass: 'pill-box' },
                    cylinder:  { icon: '🍾', label: 'Bottle / Cylinder',    pill: '3D Cylinder',   pillClass: 'pill-cylinder' },
                    person:    { icon: '🧍', label: 'Person / Human',       pill: '3D Volume',     pillClass: 'pill-depth' },
                    vehicle:   { icon: '🚗', label: 'Vehicle',              pill: '3D Volume',     pillClass: 'pill-depth' },
                    furniture: { icon: '🪑', label: 'Furniture',            pill: '3D Volume',     pillClass: 'pill-depth' },
                    generic:   { icon: '📦', label: 'Generic Object',       pill: '3D Volume',     pillClass: 'pill-depth' },
                };

                const meta = CATEGORY_META[data.category] || CATEGORY_META.generic;
                const iconEl = document.getElementById('detection-icon');
                const labelEl = document.getElementById('detection-label');
                const modePill = document.getElementById('detection-mode');

                if (iconEl) iconEl.textContent = meta.icon;
                if (labelEl) labelEl.textContent = `Detected: ${meta.label}`;
                if (modePill) {
                    modePill.textContent = meta.pill;
                    modePill.className = `detection-mode-pill ${meta.pillClass}`;
                }
                if (detectionBar) detectionBar.classList.remove('hidden');

                const dlGlb = document.getElementById('dl-glb');
                const dlZip = document.getElementById('dl-zip');
                const dlGroup = document.getElementById('download-group');
                if (dlGlb) dlGlb.href = data.glb_url;
                if (dlZip) dlZip.href = data.zip_url;
                if (dlGroup) dlGroup.classList.remove('hidden');

                const gltfLoader = new THREE.GLTFLoader();
                gltfLoader.load(data.glb_url, (gltf) => {
                    if (currentMesh) scene.remove(currentMesh);
                    currentMesh = gltf.scene;

                    currentMesh.traverse((child) => {
                        if (child.isMesh) {
                            child.material.side = THREE.DoubleSide;
                            child.material.needsUpdate = true;
                        }
                    });

                    const box = new THREE.Box3().setFromObject(currentMesh);
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3());
                    currentMesh.position.sub(center);

                    const maxDim = Math.max(size.x, size.y, size.z);
                    const fov = camera.fov * (Math.PI / 180);
                    let cameraZ = Math.abs(maxDim / (2 * Math.tan(fov / 2))) * 1.5;
                    camera.position.set(0, 0, Math.max(cameraZ, 2.5));
                    camera.lookAt(0, 0, 0);
                    if (controls) {
                        controls.target.set(0, 0, 0);
                        controls.update();
                    }

                    scene.add(currentMesh);
                });
            } catch (err) {
                if (loader) loader.classList.add('hidden');
                if (badge) badge.textContent = 'Error';
                console.error('Conversion failed:', err);
            }
        });
    }
});
