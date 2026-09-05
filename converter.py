import io
import zipfile
import numpy as np
import cv2
import torch
import trimesh
from PIL import Image
from scipy import ndimage
from skimage.measure import marching_cubes
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from rembg import remove, new_session

# ---------------------------------------------------------------------------
# Depth model candidates — tried in order, best quality first.
# Falls back to whatever is already cached if network is unavailable.
# ---------------------------------------------------------------------------
DEPTH_MODEL_CANDIDATES = [
    "depth-anything/Depth-Anything-V2-Large-hf",   # best quality ~1.3 GB
    "depth-anything/Depth-Anything-V2-Base-hf",    # good quality ~390 MB
    "depth-anything/Depth-Anything-V2-Small-hf",   # fast, cached fallback
]

# ---------------------------------------------------------------------------
# Geometry strategy lookup
# ---------------------------------------------------------------------------
CATEGORY_STRATEGY = {
    "ball":     "sphere",
    "cylinder": "cylinder",
    "generic":  "volumetric",   # true 3D-printer-style solid
}


class ImageTo3DConverter:
    def __init__(self):
        self.depth_processor = None
        self.depth_model = None
        self.depth_model_name = None
        self.rembg_session = None

    # ------------------------------------------------------------------
    # Model loading — tries best quality, falls back gracefully
    # ------------------------------------------------------------------
    def _load_depth_model(self):
        if self.depth_model is not None:
            return

        # Phase 1: Check if any candidate model is ALREADY cached locally
        for model_name in DEPTH_MODEL_CANDIDATES:
            try:
                self.depth_processor = AutoImageProcessor.from_pretrained(
                    model_name, local_files_only=True
                )
                self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                    model_name, local_files_only=True
                )
                self.depth_model.eval()
                self.depth_model_name = model_name
                print(f"✅ Loaded cached depth model instantly: {model_name}")
                return
            except Exception:
                continue

        # Phase 2: If none are cached locally, download the fast Small model (~90MB) or Base model (~390MB)
        # Avoid downloading 1.3GB Large model automatically.
        download_candidates = [
            "depth-anything/Depth-Anything-V2-Small-hf",
            "depth-anything/Depth-Anything-V2-Base-hf",
            "depth-anything/Depth-Anything-V2-Large-hf",
        ]

        for model_name in download_candidates:
            try:
                print(f"⏳ Downloading depth model (one-time setup): {model_name}...")
                self.depth_processor = AutoImageProcessor.from_pretrained(model_name)
                self.depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)
                self.depth_model.eval()
                self.depth_model_name = model_name
                print(f"✅ Depth model downloaded and cached: {model_name}")
                return
            except Exception as e:
                print(f"⚠️ Could not download {model_name}: {e}")
                continue

        raise RuntimeError(
            "Could not load or download any depth model. Check your internet connection."
        )

    # ------------------------------------------------------------------
    # Offline shape-based object classifier — zero downloads, zero network
    # ------------------------------------------------------------------
    def classify_object(self, image: Image.Image) -> tuple[str, str]:
        """
        Classify the main object using OpenCV contour shape analysis.
        Returns (category, geometry_strategy).
        """
        rgb   = np.array(image.convert("RGB"))
        h_img, w_img = rgb.shape[:2]
        gray  = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        _, thresh     = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
        _, thresh_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        def _largest_contour(mask):
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return max(cnts, key=cv2.contourArea) if cnts else None

        c_norm = _largest_contour(thresh)
        c_inv  = _largest_contour(thresh_inv)

        if c_norm is None and c_inv is None:
            return "generic", "volumetric"
        if c_norm is None:
            contour = c_inv
        elif c_inv is None:
            contour = c_norm
        else:
            contour = c_norm if cv2.contourArea(c_norm) >= cv2.contourArea(c_inv) else c_inv

        area      = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < h_img * w_img * 0.05:
            return "generic", "volumetric"

        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        _, _, bw, bh = cv2.boundingRect(contour)
        bbox_aspect  = bw / max(bh, 1)
        extent       = area / max(bw * bh, 1)
        hull_area    = cv2.contourArea(cv2.convexHull(contour))
        solidity     = area / max(hull_area, 1)

        # BALL
        if circularity > 0.60 and 0.65 < bbox_aspect < 1.55 and extent > 0.50:
            return "ball", "sphere"

        # CYLINDER
        if bbox_aspect < 0.60 and circularity > 0.25 and solidity > 0.70:
            return "cylinder", "cylinder"

        # Everything else → volumetric 3D reconstruction
        return "generic", "volumetric"

    # ------------------------------------------------------------------
    # Depth estimation
    # ------------------------------------------------------------------
    def get_depth_map(self, image: Image.Image) -> np.ndarray:
        self._load_depth_model()
        inputs = self.depth_processor(images=image.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        depth = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
        depth_u8 = (depth * 255).astype(np.uint8)
        smoothed = cv2.bilateralFilter(depth_u8, d=9, sigmaColor=75, sigmaSpace=75)
        return smoothed.astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    # Background removal
    # ------------------------------------------------------------------
    def remove_background(self, raw_img: Image.Image) -> tuple[Image.Image, np.ndarray]:
        try:
            if self.rembg_session is None:
                self.rembg_session = new_session("u2netp")
            img_nobg = remove(raw_img, session=self.rembg_session)
        except Exception:
            img_nobg = self._isolate_foreground_opencv(raw_img)
        alpha_mask = np.array(img_nobg.split()[-1]) > 20
        return img_nobg, alpha_mask

    def _isolate_foreground_opencv(self, img: Image.Image) -> Image.Image:
        rgb  = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        return Image.fromarray(np.dstack([rgb, mask_clean]), mode="RGBA")

    # ------------------------------------------------------------------
    # Geometry builders
    # ------------------------------------------------------------------

    def _build_volumetric_mesh(
        self,
        img_nobg: Image.Image,
        alpha_mask: np.ndarray,
        depth: np.ndarray,
        depth_scale: float,
        res: int,
    ) -> trimesh.Trimesh:
        """
        TRUE volumetric reconstruction using Marching Cubes.

        Builds a solid, watertight 3D mesh — exactly like a 3D printer output:
          • Front surface follows the depth contours of the image precisely
          • Object is a solid volume, not a shell or a textured primitive
          • Silhouette edges taper naturally (no harsh cliffs)
          • Gaussian-smoothed voxel grid produces organic, rounded geometry

        This replaces ALL primitive-based approaches (box, generic depth-slab).
        """
        # Resize to working resolution
        img_resized   = img_nobg.resize((res, res), Image.Resampling.LANCZOS)
        depth_r       = cv2.resize(depth, (res, res), interpolation=cv2.INTER_LINEAR)
        mask_r        = cv2.resize(
            alpha_mask.astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST
        ) > 0

        # Zero background, apply bilateral smoothing for clean depth edges
        depth_r[~mask_r] = 0.0
        depth_smooth = cv2.bilateralFilter(
            (depth_r * 255).astype(np.uint8), d=11, sigmaColor=80, sigmaSpace=80
        ).astype(np.float32) / 255.0

        # Distance-transform taper: depth fades near the silhouette edge,
        # producing naturally rounded sides (no sharp cliff at object boundary)
        dist   = cv2.distanceTransform(mask_r.astype(np.uint8), cv2.DIST_L2, 5)
        taper  = np.clip(dist / max(dist.max() * 0.15, 1.0), 0.0, 1.0)
        depth_smooth = depth_smooth * taper

        # ── Build 3D occupancy volume ──────────────────────────────────
        # Number of depth slices controls object "thickness"
        depth_levels = 56

        # Per-pixel front face depth index [0 … depth_levels-1]
        front_d = np.clip(
            np.round(depth_smooth * (depth_levels - 1) * min(depth_scale * 4.0, 1.0)).astype(int),
            1, depth_levels - 1,
        )

        # Fixed back face (solid base ~18 % of total depth)
        back_d = max(3, int(depth_levels * 0.18))

        # Vectorised fill:  vol[row, col, z] = 1  iff  back_d ≤ z ≤ front_d[row,col]
        z_idx  = np.arange(depth_levels)                       # (D,)
        vol    = (
            mask_r[:, :, None]                                  # foreground pixels only
            & (z_idx[None, None, :] >= back_d)                 # above back face
            & (z_idx[None, None, :] <= front_d[:, :, None])    # below front face
        ).astype(np.float32)

        # Gaussian smooth the voxel grid → smooth, organic surface geometry
        vol = ndimage.gaussian_filter(vol, sigma=[1.2, 1.2, 1.5])

        # ── Marching Cubes → watertight triangulated mesh ──────────────
        verts, faces, normals, _ = marching_cubes(vol, level=0.5)

        # Normalise vertex coords to [-1, 1] in each axis
        vn = np.empty_like(verts)
        vn[:, 0] = verts[:, 0] / res          * 2.0 - 1.0   # row  → Y
        vn[:, 1] = verts[:, 1] / res          * 2.0 - 1.0   # col  → X
        vn[:, 2] = verts[:, 2] / depth_levels * 2.0 - 1.0   # z    → Z (depth)

        # ── Front-projection UV mapping ────────────────────────────────
        # Each vertex UV = where it sits on the image plane (camera POV projection)
        uv_u = np.clip((vn[:, 1] + 1.0) / 2.0,        0.0, 1.0)
        uv_v = np.clip(1.0 - (vn[:, 0] + 1.0) / 2.0,  0.0, 1.0)
        uvs  = np.column_stack([uv_u, uv_v])

        material = trimesh.visual.texture.SimpleMaterial(image=img_resized)
        mesh = trimesh.Trimesh(
            vertices=vn,
            faces=faces,
            vertex_normals=normals,
            visual=trimesh.visual.TextureVisuals(
                uv=uvs, image=img_resized, material=material
            ),
            process=True,
        )
        mesh.fix_normals()
        return mesh

    def _build_sphere_mesh(self, img_resized: Image.Image) -> trimesh.Trimesh:
        """High-subdivision UV sphere with equirectangular texture projection."""
        sphere  = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
        normals = sphere.vertex_normals
        u = 0.5 + np.arctan2(normals[:, 0], normals[:, 2]) / (2 * np.pi)
        v = 0.5 - np.arcsin(np.clip(normals[:, 1], -1, 1)) / np.pi
        uvs = np.column_stack([u, v])
        material = trimesh.visual.texture.SimpleMaterial(image=img_resized)
        sphere.visual = trimesh.visual.TextureVisuals(uv=uvs, image=img_resized, material=material)
        return sphere

    def _build_cylinder_mesh(self, img_resized: Image.Image) -> trimesh.Trimesh:
        """Cylinder with cylindrical UV wrapping."""
        cyl   = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=72)
        verts = cyl.vertices
        u = (np.arctan2(verts[:, 0], verts[:, 1]) / (2 * np.pi)) % 1.0
        v = (verts[:, 2] - verts[:, 2].min()) / (verts[:, 2].max() - verts[:, 2].min() + 1e-8)
        uvs = np.column_stack([u, v])
        material = trimesh.visual.texture.SimpleMaterial(image=img_resized)
        cyl.visual = trimesh.visual.TextureVisuals(uv=uvs, image=img_resized, material=material)
        return cyl

    # ------------------------------------------------------------------
    # Main conversion entry point
    # ------------------------------------------------------------------
    def convert(
        self,
        image_bytes: bytes,
        depth_scale: float = 0.25,
        res: int = 256,
        isolate_subject: bool = True,
    ):
        raw_img = Image.open(io.BytesIO(image_bytes))

        # 1. Classify object → geometry strategy
        category, strategy = self.classify_object(raw_img)

        # 2. Remove background
        if isolate_subject:
            img_nobg, alpha_mask = self.remove_background(raw_img)
        else:
            img_nobg   = raw_img.convert("RGBA")
            alpha_mask = np.ones((raw_img.height, raw_img.width), dtype=bool)

        # 3. Estimate depth map
        img_rgb = img_nobg.convert("RGB")
        depth   = self.get_depth_map(img_rgb)
        depth[~alpha_mask] = 0.0

        # 4. Resized texture for primitive strategies
        img_resized = img_nobg.resize((res, res), Image.Resampling.LANCZOS)

        # 5. Build mesh
        if strategy == "sphere":
            mesh = self._build_sphere_mesh(img_resized)
        elif strategy == "cylinder":
            mesh = self._build_cylinder_mesh(img_resized)
        else:
            # "volumetric" — true marching-cubes 3D solid for everything else
            mesh = self._build_volumetric_mesh(
                img_nobg, alpha_mask, depth, depth_scale, res
            )

        # 6. Export GLB
        glb_bytes = mesh.export(file_type="glb")

        # 7. Export OBJ + MTL + texture ZIP
        obj_content = mesh.export(file_type="obj")
        img_buf = io.BytesIO()
        img_resized.save(img_buf, format="PNG")

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("model.obj", obj_content)
            zf.writestr("texture.png", img_buf.getvalue())
            zf.writestr("model.mtl", "newmtl material_0\nmap_Kd texture.png\n")

        model_label = self.depth_model_name.split("/")[-1] if self.depth_model_name else "unknown"
        return glb_bytes, zip_buf.getvalue(), depth, category, strategy, model_label
