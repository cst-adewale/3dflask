import os
import time
import io
import zipfile
import requests
import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

TRIPO_API_KEY = os.getenv("TRIPO_API_KEY")
TRIPO_API_URL = "https://api.tripo3d.ai/v2/openapi"


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

        # Step 2: Create image_to_model task
        task_resp = requests.post(
            f"{TRIPO_API_URL}/task",
            headers=headers,
            json={"type": "image_to_model", "file": {"type": "png", "file_token": image_token}},
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
                model_url = output_data.get("model") or output_data.get("pbr_model")
                if model_url:
                    print(f"✅ Tripo3D 3D Model generated! Downloading from {model_url}...")
                    glb_resp = requests.get(model_url, timeout=60)
                    if glb_resp.status_code == 200:
                        return glb_resp.content
                break
            elif task_status == "failed":
                raise RuntimeError("Tripo3D task generation failed on server side.")

        raise RuntimeError("Tripo3D generation timed out.")

    def convert(
        self,
        image_bytes: bytes,
        depth_scale: float = 0.25,
        res: int = 256,
        isolate_subject: bool = True,
    ):
        raw_img = Image.open(io.BytesIO(image_bytes))
        category, strategy = self.classify_object(raw_img)

        # 1. Tripo3D Generative AI 3D Reconstruction
        tripo_glb = self._generate_tripo3d(image_bytes)

        # 2. Fast OpenCV depth-map visualization for UI preview
        gray = cv2.cvtColor(np.array(raw_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        depth_map = (gray.astype(np.float32) / 255.0)

        # 3. Downloadable ZIP bundle
        img_resized = raw_img.resize((res, res), Image.Resampling.LANCZOS)
        img_buf = io.BytesIO()
        img_resized.save(img_buf, format="PNG")

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("model.glb", tripo_glb)
            zf.writestr("texture.png", img_buf.getvalue())

        return tripo_glb, zip_buf.getvalue(), depth_map, category, "Generative 3D (Tripo3D)", "Tripo3D-V2"
