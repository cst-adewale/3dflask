let scene, camera, renderer, controls, currentMesh;

function init3D() {
    const container = document.getElementById('canvas-container');
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x090d16);

    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(0, 0, 3);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.outputEncoding = THREE.sRGBEncoding;
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // Ambient & Directional Lights
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 1.2);
    scene.add(hemiLight);
    
    const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight1.position.set(2, 4, 3);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
    dirLight2.position.set(-2, -2, -3);
    scene.add(dirLight2);

    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });

    animate();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// UI Handlers
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('image-input');
const imgPreview = document.getElementById('image-preview');
const dropzoneText = document.getElementById('dropzone-text');

dropzone.onclick = () => fileInput.click();
fileInput.onchange = (e) => {
    if (e.target.files[0]) {
        const url = URL.createObjectURL(e.target.files[0]);
        imgPreview.src = url;
        imgPreview.classList.remove('hidden');
        dropzoneText.classList.add('hidden');
    }
};

document.getElementById('depth_scale').oninput = (e) => document.getElementById('scale-val').textContent = e.target.value;
document.getElementById('resolution').oninput = (e) => document.getElementById('res-val').textContent = e.target.value;

document.getElementById('convert-form').onsubmit = async (e) => {
    e.preventDefault();
    if (!fileInput.files[0]) return;

    const loader = document.getElementById('loader');
    const badge = document.getElementById('status-badge');
    const detectionBar = document.getElementById('detection-bar');
    loader.classList.remove('hidden');
    badge.textContent = 'Classifying & generating...';
    detectionBar.classList.add('hidden');

    const formData = new FormData();
    formData.append('image', fileInput.files[0]);
    formData.append('depth_scale', document.getElementById('depth_scale').value);
    formData.append('resolution', document.getElementById('resolution').value);
    formData.append('isolate_subject', document.getElementById('isolate_subject').checked);

    const res = await fetch('/api/convert', { method: 'POST', body: formData });
    const data = await res.json();

    loader.classList.add('hidden');
    badge.textContent = 'Rendered';

    // --- Detection Banner ---
    const CATEGORY_META = {
        ball:      { icon: '🏀', label: 'Ball / Sports Ball',   pill: 'Sphere Mode',   pillClass: 'pill-sphere' },
        door:      { icon: '🚪', label: 'Door / Window / Panel', pill: 'Box Mode',      pillClass: 'pill-box' },
        cylinder:  { icon: '🍾', label: 'Bottle / Cylinder',    pill: 'Cylinder Mode', pillClass: 'pill-cylinder' },
        person:    { icon: '🧍', label: 'Person / Human',       pill: 'Depth Mode',    pillClass: 'pill-depth' },
        vehicle:   { icon: '🚗', label: 'Vehicle',              pill: 'Depth Mode',    pillClass: 'pill-depth' },
        furniture: { icon: '🪑', label: 'Furniture',            pill: 'Depth Mode',    pillClass: 'pill-depth' },
        generic:   { icon: '📦', label: 'Generic Object',       pill: 'Depth Mode',    pillClass: 'pill-depth' },
    };
    const meta = CATEGORY_META[data.category] || CATEGORY_META.generic;
    document.getElementById('detection-icon').textContent = meta.icon;
    document.getElementById('detection-label').textContent = `Detected: ${meta.label}`;
    const modePill = document.getElementById('detection-mode');
    modePill.textContent = meta.pill;
    modePill.className = `detection-mode-pill ${meta.pillClass}`;
    detectionBar.classList.remove('hidden');

    // Show Depth Map
    const depthImg = document.getElementById('depth-preview');
    depthImg.src = data.depth_map;
    depthImg.style.display = 'block';

    // Set Download Links
    document.getElementById('dl-glb').href = data.glb_url;
    document.getElementById('dl-zip').href = data.zip_url;
    document.getElementById('download-group').classList.remove('hidden');

    // Load 3D Model into Viewport with auto-framing and double-sided material
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

        // Auto-center bounding box
        const box = new THREE.Box3().setFromObject(currentMesh);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        currentMesh.position.sub(center);

        const maxDim = Math.max(size.x, size.y, size.z);
        const fov = camera.fov * (Math.PI / 180);
        let cameraZ = Math.abs(maxDim / (2 * Math.tan(fov / 2))) * 1.5;
        camera.position.set(0, 0, Math.max(cameraZ, 2.5));
        camera.lookAt(0, 0, 0);
        controls.target.set(0, 0, 0);
        controls.update();

        scene.add(currentMesh);
    });
};

window.onload = init3D;
