import os
import time
import io
import zipfile
import requests
import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

load_dotenv()

TRIPO_API_KEY = os.getenv("TRIPO_API_KEY")
TRIPO_API_URL = "https://api.tripo3d.ai/v2/openapi"

def run_with_timeout(func, args=(), kwargs=None, timeout=35):
    if kwargs is None:
        kwargs = {}
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)



class ImageTo3DConverter:
    def __init__(self):
        pass

    def classify_object(self, image: Image.Image) -> tuple[str, str]:
        """
        Classify the main object using OpenCV contour shape analysis.
        Returns (category, geometry_strategy).
        """
        rgb = np.array(image.convert("RGB"))
        h_img, w_img = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        def _largest_contour(mask):
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return max(cnts, key=cv2.contourArea) if cnts else None

        c_norm = _largest_contour(thresh)
        c_inv = _largest_contour(thresh_inv)

        if c_norm is None and c_inv is None:
            return "generic", "generative"
        if c_norm is None:
            contour = c_inv
        elif c_inv is None:
            contour = c_norm
        else:
            contour = c_norm if cv2.contourArea(c_norm) >= cv2.contourArea(c_inv) else c_inv

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < h_img * w_img * 0.05:
            return "generic", "generative"

        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        _, _, bw, bh = cv2.boundingRect(contour)
        bbox_aspect = bw / max(bh, 1)
        extent = area / max(bw * bh, 1)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / max(hull_area, 1)

        if circularity > 0.60 and 0.65 < bbox_aspect < 1.55 and extent > 0.50:
            return "ball", "generative"
        if bbox_aspect < 0.60 and circularity > 0.25 and solidity > 0.70:
            return "cylinder", "generative"

        return "generic", "generative"

    def _generate_tripo3d(self, image_bytes: bytes) -> bytes:
        """
        Sends image to Tripo3D API for state-of-the-art 360 generative 3D mesh reconstruction.
        """
        if not TRIPO_API_KEY:
            raise RuntimeError("TRIPO_API_KEY environment variable is not configured.")

        headers = {"Authorization": f"Bearer {TRIPO_API_KEY}"}

        print("🚀 Sending image to Tripo3D Generative AI API...")
        # Step 1: Upload image file to Tripo3D storage
        upload_resp = requests.post(
            f"{TRIPO_API_URL}/upload",
            headers=headers,
            files={"file": ("image.png", image_bytes, "image/png")},
            timeout=30,
        )
        if upload_resp.status_code != 200:
            raise RuntimeError(f"Tripo3D Upload failed ({upload_resp.status_code}): {upload_resp.text}")

        image_token = upload_resp.json().get("data", {}).get("image_token")
        if not image_token:
            raise RuntimeError("Tripo3D did not return an image_token.")

        # Step 2: Create image_to_model task with PBR material generation enabled
        task_resp = requests.post(
            f"{TRIPO_API_URL}/task",
            headers=headers,
            json={
                "type": "image_to_model",
                "file": {"type": "png", "file_token": image_token},
                "pbr": True
            },
            timeout=30,
        )
        if task_resp.status_code != 200:
            raise RuntimeError(f"Tripo3D Task creation failed ({task_resp.status_code}): {task_resp.text}")

        task_id = task_resp.json().get("data", {}).get("task_id")
        if not task_id:
            raise RuntimeError("Tripo3D task_id missing.")

        print(f"⌛ Tripo3D Task submitted (ID: {task_id}). Waiting for 3D model generation...")

        # Step 3: Poll task status until complete (max 120 sec)
        for _ in range(60):
            time.sleep(2)
            status_resp = requests.get(f"{TRIPO_API_URL}/task/{task_id}", headers=headers, timeout=15)
            if status_resp.status_code != 200:
                continue

            status_data = status_resp.json().get("data", {})
            task_status = status_data.get("status")

            if task_status == "success":
                output_data = status_data.get("output", {})
                # Prioritize PBR textured model over untextured draft model
                model_url = output_data.get("pbr_model") or output_data.get("model") or output_data.get("base_model")
                if model_url:
                    print(f"✅ Tripo3D Textured 3D Model generated! Downloading from {model_url}...")
                    glb_resp = requests.get(model_url, timeout=60)
                    if glb_resp.status_code == 200:
                        return glb_resp.content
                break
            elif task_status == "failed":
                raise RuntimeError("Tripo3D task generation failed on server side.")

        raise RuntimeError("Tripo3D generation timed out.")

    def generate_local_3d(self, image_bytes: bytes, depth_scale: float = 0.25, res: int = 256) -> bytes:
        """
        Generates a 3D textured GLB mesh locally using depth-map displacement and UV mapping.
        100% Free, instant, and offline.
        """
        import trimesh

        raw_img = Image.open(io.BytesIO(image_bytes))
        img_res = raw_img.resize((res, res), Image.Resampling.LANCZOS)
        rgb = np.array(img_res.convert("RGB"))
        alpha = np.array(img_res.getchannel("A")) if "A" in raw_img.getbands() else np.full((res, res), 255, dtype=np.uint8)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        alpha_mask = (alpha > 30).astype(np.float32)

        x = np.linspace(-1.0, 1.0, res, dtype=np.float32)
        y = np.linspace(1.0, -1.0, res, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        z_front = gray_blur * depth_scale * 2.0 * alpha_mask
        z_back = -0.05 * alpha_mask

        u = np.linspace(0.0, 1.0, res, dtype=np.float32)
        v = np.linspace(1.0, 0.0, res, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)

        v_front = np.stack([xx, yy, z_front], axis=-1).reshape(-1, 3)
        v_back = np.stack([xx, yy, z_back], axis=-1).reshape(-1, 3)
        vertices = np.concatenate([v_front, v_back], axis=0)

        uv_front = np.stack([uu, vv], axis=-1).reshape(-1, 2)
        uv_back = np.stack([uu, vv], axis=-1).reshape(-1, 2)
        uvs = np.concatenate([uv_front, uv_back], axis=0)

        num_verts_per_side = res * res
        faces = []

        for r in range(res - 1):
            for c in range(res - 1):
                i0 = r * res + c
                i1 = i0 + 1
                i2 = (r + 1) * res + c
                i3 = i2 + 1

                if alpha_mask[r, c] > 0 or alpha_mask[r + 1, c] > 0 or alpha_mask[r, c + 1] > 0:
                    faces.append([i0, i2, i1])
                    faces.append([i1, i2, i3])

                    b0 = i0 + num_verts_per_side
                    b1 = i1 + num_verts_per_side
                    b2 = i2 + num_verts_per_side
                    b3 = i3 + num_verts_per_side
                    faces.append([b0, b1, b2])
                    faces.append([b1, b3, b2])

        faces = np.array(faces, dtype=np.int32)
        material = trimesh.visual.texture.SimpleMaterial(image=img_res.convert("RGB"))
        visuals = trimesh.visual.TextureVisuals(uv=uvs, material=material)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visuals, validate=False)
        return mesh.export(file_type="glb")

    def _generate_wonder3d(self, image_bytes: bytes) -> bytes:
        """
        Generates 3D mesh using Wonder3D / TripoSR single-image multi-view reconstruction via HuggingFace Gradio API.
        """
        import tempfile
        from gradio_client import Client, handle_file

        print("🚀 Sending image to Wonder3D AI Engine...")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            client = Client("stabilityai/TripoSR")
            result = client.predict(
                image_path=handle_file(tmp_path),
                do_remove_background=True,
                foreground_ratio=0.85,
                mc_resolution=256,
                api_name="/generate_mesh"
            )
            glb_path = result[0] if isinstance(result, (tuple, list)) else result
            with open(glb_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _generate_instantmesh(self, image_bytes: bytes) -> bytes:
        """
        Generates a high-quality 3D GLB using InstantMesh (TencentARC) via HuggingFace Gradio.
        3-step pipeline: preprocess -> generate_mvs -> make3d
        """
        import tempfile
        from gradio_client import Client, handle_file

        print("[*] Sending image to InstantMesh AI Engine (TencentARC)...")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            client = Client("TencentARC/InstantMesh")

            # Step 1: Preprocess (background removal + centering)
            print("[*] InstantMesh step 1/3: preprocessing image...")
            preprocessed = client.predict(
                input_image=handle_file(tmp_path),
                do_remove_background=True,
                api_name="/preprocess"
            )
            preprocessed_path = preprocessed["path"] if isinstance(preprocessed, dict) else preprocessed

            # Step 2: Generate multi-view images
            print("[*] InstantMesh step 2/3: generating multi-view images...")
            client.predict(
                input_image=handle_file(preprocessed_path),
                sample_steps=75,
                sample_seed=42,
                api_name="/generate_mvs"
            )

            # Step 3: Reconstruct 3D mesh from multi-views
            print("[*] InstantMesh step 3/3: reconstructing 3D mesh...")
            result = client.predict(api_name="/make3d")

            # result is (obj_path, glb_path)
            glb_path = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else result[0]
            with open(glb_path, "rb") as f:
                return f.read()
        finally:
            for p in [tmp_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    def convert(
        self,
        image_bytes: bytes,
        depth_scale: float = 0.25,
        res: int = 256,
        isolate_subject: bool = True,
        engine: str = "auto",
    ):
        raw_img = Image.open(io.BytesIO(image_bytes))
        category, strategy = self.classify_object(raw_img)

        used_engine = "Local 3D Engine (Free)"
        strategy_label = "Local Heightmap Extrusion"
        glb_bytes = None

        if engine == "local":
            glb_bytes = self.generate_local_3d(image_bytes, depth_scale, res)
            used_engine = "Local 3D Engine (Free)"
            strategy_label = "Local Heightmap Extrusion"
        elif engine == "instantmesh":
            try:
                glb_bytes = run_with_timeout(self._generate_instantmesh, args=(image_bytes,), timeout=30)
                used_engine = "InstantMesh AI Engine"
                strategy_label = "InstantMesh Multi-View Mesh"
            except Exception as err:
                print(f"[!] InstantMesh note ({err}). Using Free Local Engine.")
                glb_bytes = self.generate_local_3d(image_bytes, depth_scale, res)
                used_engine = "InstantMesh (Local Fallback)"
                strategy_label = "Local Heightmap Extrusion"
        elif engine == "wonder3d":
            try:
                glb_bytes = run_with_timeout(self._generate_wonder3d, args=(image_bytes,), timeout=30)
                used_engine = "Wonder3D AI Engine"
                strategy_label = "Wonder3D Multi-View Mesh"
            except Exception as err:
                print(f"⚠️ Wonder3D AI Engine note ({err}). Using Free Local Engine.")
                glb_bytes = self.generate_local_3d(image_bytes, depth_scale, res)
                used_engine = "Wonder3D (Local Fallback)"
                strategy_label = "Local Heightmap Extrusion"
        elif engine == "tripo":
            if TRIPO_API_KEY and TRIPO_API_KEY.strip():
                try:
                    glb_bytes = run_with_timeout(self._generate_tripo3d, args=(image_bytes,), timeout=30)
                    used_engine = "Tripo3D Cloud AI"
                    strategy_label = "Generative 3D (Tripo3D)"
                except Exception as err:
                    print(f"⚠️ Tripo3D unavailable ({err}). Falling back to Free Local Engine.")
                    glb_bytes = self.generate_local_3d(image_bytes, depth_scale, res)
                    used_engine = "Tripo3D (Local Fallback)"
                    strategy_label = "Local Heightmap Extrusion"
            else:
                glb_bytes = self.generate_local_3d(image_bytes, depth_scale, res)
                used_engine = "Local 3D Engine (Free)"
                strategy_label = "Local Heightmap Extrusion"
        else:  # auto — strongest to weakest cascade: Tripo3D -> InstantMesh -> Wonder3D -> Free Local
            if TRIPO_API_KEY and TRIPO_API_KEY.strip():
                try:
                    glb_bytes = run_with_timeout(self._generate_tripo3d, args=(image_bytes,), timeout=30)
                    used_engine = "Tripo3D Cloud AI"
                    strategy_label = "Generative 3D (Tripo3D)"
                except Exception:
                    pass

            if not glb_bytes:
                try:
                    glb_bytes = run_with_timeout(self._generate_instantmesh, args=(image_bytes,), timeout=25)
                    used_engine = "InstantMesh AI Engine"
                    strategy_label = "InstantMesh Multi-View Mesh"
                except Exception:
                    try:
                        glb_bytes = run_with_timeout(self._generate_wonder3d, args=(image_bytes,), timeout=25)
                        used_engine = "Wonder3D AI Engine"
                        strategy_label = "Wonder3D Multi-View Mesh"
                    except Exception:
                        glb_bytes = self.generate_local_3d(image_bytes, depth_scale, res)
                        used_engine = "Local 3D Engine (Free)"
                        strategy_label = "Local Heightmap Extrusion"

        # OpenCV depth-map visualization
        gray = cv2.cvtColor(np.array(raw_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        depth_map = gray.astype(np.float32) / 255.0

        # ZIP bundle containing GLB, OBJ, MTL, and texture
        img_resized = raw_img.resize((res, res), Image.Resampling.LANCZOS)
        img_buf = io.BytesIO()
        img_resized.save(img_buf, format="PNG")

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("model.glb", glb_bytes)
            zf.writestr("texture.png", img_buf.getvalue())

            try:
                import trimesh
                scene_or_mesh = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
                if isinstance(scene_or_mesh, trimesh.Scene):
                    mesh = trimesh.util.concatenate(scene_or_mesh.dump())
                else:
                    mesh = scene_or_mesh

                obj_data = trimesh.exchange.obj.export_obj(mesh, include_normals=True, include_color=True, include_texture=True)
                if isinstance(obj_data, dict):
                    for filename, file_content in obj_data.items():
                        zf.writestr(filename, file_content)
                elif isinstance(obj_data, str):
                    zf.writestr("model.obj", obj_data)
            except Exception as e:
                print(f"OBJ export fallback note: {e}")

        return glb_bytes, zip_buf.getvalue(), depth_map, category, strategy_label, used_engine

