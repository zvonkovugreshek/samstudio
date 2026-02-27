# SAM 3 Segmentation App — Ultralytics Edition
# Streamlit Cloud Deployment
#
# Deploy:
#   1. Push this repo to GitHub
#   2. Go to share.streamlit.io → New app → select repo
#   3. Set main file path to: app.py
#   4. Add HF_TOKEN in Settings → Secrets:
#        HF_TOKEN = "hf_your_token_here"

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import io
import tempfile

# --- App Configuration ---
st.set_page_config(
    page_title="SAM 3 Studio",
    page_icon="🎯",
    layout="wide",
)


# ─── Model Weight Download ───────────────────────────────────────────────────

def ensure_sam3_weights(model_path="sam3.pt"):
    """
    Downloads sam3.pt from HuggingFace if not already present.
    Uses HF_TOKEN from Streamlit secrets (cloud) or environment variable (local).
    """
    if os.path.exists(model_path):
        return model_path

    hf_token = None
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except (FileNotFoundError, KeyError):
        hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        st.error(
            "**HF_TOKEN not found.** SAM 3 weights are gated on HuggingFace.\n\n"
            "**For Streamlit Cloud:** Go to Settings → Secrets and add:\n"
            "```\nHF_TOKEN = \"hf_your_token_here\"\n```\n\n"
            "**For local use:** Set the `HF_TOKEN` environment variable or create "
            "`.streamlit/secrets.toml` with `HF_TOKEN = \"hf_...\"`"
        )
        st.stop()

    with st.spinner("Downloading SAM 3 weights from HuggingFace (first run only)..."):
        try:
            from huggingface_hub import login, hf_hub_download

            login(token=hf_token)
            hf_hub_download(
                repo_id="facebook/sam3",
                filename="sam3.pt",
                local_dir=".",
            )
            st.success("SAM 3 weights downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download SAM 3 weights: {e}")
            st.stop()

    return model_path


# ─── Model Loading (Cached) ─────────────────────────────────────────────────

@st.cache_resource
def load_semantic_predictor(model_path="sam3.pt", conf=0.25):
    """SAM3SemanticPredictor for text-based concept segmentation."""
    from ultralytics.models.sam import SAM3SemanticPredictor

    use_half = _has_cuda()
    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        model=model_path,
        half=use_half,
        save=False,
        verbose=False,
    )
    return SAM3SemanticPredictor(overrides=overrides)


@st.cache_resource
def load_sam_model(model_path="sam3.pt"):
    """SAM model for interactive point/box prompts (SAM2-compatible)."""
    from ultralytics import SAM
    return SAM(model_path)


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ─── Point Drawing Helper ────────────────────────────────────────────────────

def draw_points_on_image(image_pil, points, radius=8):
    """Draw red circles on the image at the given point coordinates."""
    from PIL import ImageDraw
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)
    for i, (x, y) in enumerate(points):
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill="#FF6B35", outline="white", width=2,
        )
        # Draw number label
        draw.text((x + radius + 3, y - radius), str(i + 1), fill="white")
    return img


# ─── Visualization Helpers ───────────────────────────────────────────────────

def overlay_masks_on_image(image_np, masks_np, alpha=0.5):
    """Overlays binary masks on an image with random colors."""
    if masks_np is None or len(masks_np) == 0:
        return image_np

    vis = image_np.copy()

    if masks_np.ndim == 2:
        masks_np = masks_np[np.newaxis, :, :]

    np.random.seed(42)
    colors = np.random.randint(50, 230, (len(masks_np), 3), dtype=np.uint8)

    for i, mask in enumerate(masks_np):
        color = colors[i]
        where = mask > 0
        overlay = vis.copy()
        for c in range(3):
            overlay[..., c][where] = (
                alpha * color[c] + (1 - alpha) * overlay[..., c][where]
            ).astype(np.uint8)
        vis = overlay

    return vis


def masks_to_single_binary(masks_np):
    """Flatten N masks into a single combined binary mask."""
    if masks_np is None:
        return None
    if masks_np.ndim == 2:
        return (masks_np > 0).astype(np.uint8)
    return (np.max(masks_np, axis=0) > 0).astype(np.uint8)


def create_cutout(image_np, masks_np):
    """Creates an RGBA cutout of the segmented area, cropped to bounding box."""
    combined = masks_to_single_binary(masks_np)
    if combined is None:
        return Image.fromarray(image_np)
    return _cutout_from_binary(image_np, combined)


def _cutout_from_binary(image_np, binary_mask):
    """Creates an RGBA cutout from a single (H, W) binary mask, cropped to bbox."""
    if image_np.shape[2] == 3:
        rgba = np.dstack((image_np, np.full(image_np.shape[:2], 255, dtype=np.uint8)))
    else:
        rgba = image_np.copy()

    rgba[:, :, 3] = np.where(binary_mask > 0, 255, 0).astype(np.uint8)
    img = Image.fromarray(rgba)

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


def render_results(image_pil, image_np, masks_np, caption="Segmented Result", prefix="seg"):
    """Render side-by-side comparison, combined downloads, and per-segment downloads."""
    if masks_np.ndim == 2:
        masks_np = masks_np[np.newaxis, :, :]

    n_masks = len(masks_np)

    # Side-by-side preview
    c1, c2 = st.columns(2)
    c1.image(image_pil, caption="Original")

    overlaid = overlay_masks_on_image(image_np, masks_np)
    c2.image(overlaid, caption=caption)

    # Combined downloads (all segments merged)
    st.markdown("**All segments combined:**")
    combined = masks_to_single_binary(masks_np)
    col_a, col_b = st.columns(2)

    mask_img = Image.fromarray((combined * 255).astype(np.uint8))
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    col_a.download_button(
        "⬇️ Download Combined Mask", data=buf.getvalue(),
        file_name=f"{prefix}_mask_all.png", mime="image/png",
        key=f"{prefix}_dl_mask_all",
    )

    cutout = create_cutout(image_np, masks_np)
    buf_c = io.BytesIO()
    cutout.save(buf_c, format="PNG")
    col_b.download_button(
        "⬇️ Download Combined Cutout", data=buf_c.getvalue(),
        file_name=f"{prefix}_cutout_all.png", mime="image/png",
        key=f"{prefix}_dl_cutout_all",
    )

    # Individual segment downloads (only if more than 1 mask)
    if n_masks > 1:
        with st.expander(f"📦 Download individual segments ({n_masks} found)", expanded=False):
            np.random.seed(42)
            colors = np.random.randint(50, 230, (n_masks, 3), dtype=np.uint8)

            for i, mask in enumerate(masks_np):
                color = colors[i]
                color_hex = "#{:02x}{:02x}{:02x}".format(*color)
                single_binary = (mask > 0).astype(np.uint8)

                st.markdown(
                    f"**Segment {i + 1}** "
                    f'<span style="color:{color_hex}">■</span> '
                    f"({int(single_binary.sum())} px)",
                    unsafe_allow_html=True,
                )

                sc1, sc2, sc3 = st.columns([2, 1, 1])

                single_overlay = overlay_masks_on_image(
                    image_np, single_binary[np.newaxis, :, :]
                )
                sc1.image(single_overlay, use_container_width=True)

                seg_mask_img = Image.fromarray((single_binary * 255).astype(np.uint8))
                buf_m = io.BytesIO()
                seg_mask_img.save(buf_m, format="PNG")
                sc2.download_button(
                    f"⬇️ Mask #{i + 1}",
                    data=buf_m.getvalue(),
                    file_name=f"{prefix}_mask_{i + 1}.png",
                    mime="image/png",
                    key=f"{prefix}_dl_mask_{i}",
                )

                seg_cutout = _cutout_from_binary(image_np, single_binary)
                buf_co = io.BytesIO()
                seg_cutout.save(buf_co, format="PNG")
                sc3.download_button(
                    f"⬇️ Cutout #{i + 1}",
                    data=buf_co.getvalue(),
                    file_name=f"{prefix}_cutout_{i + 1}.png",
                    mime="image/png",
                    key=f"{prefix}_dl_cutout_{i}",
                )

                if i < n_masks - 1:
                    st.divider()


# ─── Result Extraction ───────────────────────────────────────────────────────

def extract_masks_from_results(results):
    """Extract numpy masks (N, H, W) from Ultralytics Results objects."""
    if results is None:
        return None
    if isinstance(results, list):
        for r in results:
            if r.masks is not None:
                return r.masks.data.cpu().numpy()
    else:
        if results.masks is not None:
            return results.masks.data.cpu().numpy()
    return None


def extract_masks_from_semantic(masks_tensor, boxes_tensor):
    """Extract masks from SAM3SemanticPredictor output."""
    if masks_tensor is None:
        return None
    masks_np = masks_tensor.cpu().numpy()
    if masks_np.ndim == 2:
        masks_np = masks_np[np.newaxis, :, :]
    return masks_np


# ─── Save Image to Temp ─────────────────────────────────────────────────────

def save_image_to_temp(image_pil):
    """Save PIL image to a temp JPEG file and return path."""
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "input.jpg")
    image_pil.save(tmp_path, format="JPEG", quality=95)
    return tmp_path, tmp_dir


# ─── Main App ────────────────────────────────────────────────────────────────

def main():
    st.title("SAM 3 Studio")
    st.markdown(
        "Segment images using **Text Prompts** or **Interactive Points**.  "
        "Created by Zvonko Vugreshek from [Pixolid UG](https://pixolid.de). Powered by SAM 3"
    )

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    mode = st.sidebar.radio(
        "Input Mode",
        ["Interactive Point", "Text Prompt"],
        index=0,
    )

    confidence = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, 0.4, 0.05,
    )

    overlay_alpha = st.sidebar.slider(
        "Overlay Opacity", 0.1, 0.9, 0.5, 0.1,
    )

    # Model
    model_path = ensure_sam3_weights()

    # Upload
    uploaded_file = st.file_uploader(
        "Upload Target Image", type=["png", "jpg", "jpeg", "webp"]
    )

    if not uploaded_file:
        st.info("👆 Upload an image to get started.")
        return

    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)

    # ── INTERACTIVE POINT MODE ──
    if mode == "Interactive Point":
        st.info("☝️ Click on the image to add points, then press **Segment Points**.")

        from streamlit_image_coordinates import streamlit_image_coordinates

        w, h = image_pil.size

        # Initialize session state for points
        if "click_points" not in st.session_state:
            st.session_state.click_points = []
        if "interactive_result" not in st.session_state:
            st.session_state.interactive_result = None

        # Draw existing points on image for display
        display_img = draw_points_on_image(image_pil, st.session_state.click_points)

        # Resize for display (keep aspect ratio, max 700px wide)
        display_width = min(700, w)
        display_height = int(h * (display_width / w))
        display_img_resized = display_img.resize((display_width, display_height), Image.LANCZOS)

        # Clickable image
        coords = streamlit_image_coordinates(
            display_img_resized,
            key="img_coords",
        )

        # Handle new click
        if coords is not None:
            # Scale from display coords back to original image coords
            scale_x = w / display_width
            scale_y = h / display_height
            real_x = coords["x"] * scale_x
            real_y = coords["y"] * scale_y

            new_point = [real_x, real_y]

            # Only add if it's a new click (not a duplicate from rerun)
            if (len(st.session_state.click_points) == 0 or
                st.session_state.click_points[-1] != new_point):
                st.session_state.click_points.append(new_point)
                st.rerun()

        # Show point count and controls
        n_pts = len(st.session_state.click_points)
        if n_pts > 0:
            col_info, col_clear = st.columns([3, 1])
            col_info.caption(f"**{n_pts}** point(s) selected")
            if col_clear.button("❌ Clear Points"):
                st.session_state.click_points = []
                st.session_state.interactive_result = None
                st.rerun()

            if st.button("🔍 Segment Points", type="primary"):
                with st.spinner("Segmenting..."):
                    try:
                        tmp_path, tmp_dir = save_image_to_temp(image_pil)

                        points = st.session_state.click_points
                        labels = [1] * len(points)  # All foreground

                        model = load_sam_model(model_path)
                        results = model.predict(
                            source=tmp_path,
                            points=points,
                            labels=labels,
                            verbose=False,
                        )

                        masks_np = extract_masks_from_results(results)

                        if masks_np is not None and len(masks_np) > 0:
                            st.session_state.interactive_result = {
                                "image_pil": image_pil,
                                "image_np": image_np,
                                "masks": masks_np,
                            }
                        else:
                            st.session_state.interactive_result = None
                            st.warning("No segments found at the selected point(s).")

                        try:
                            os.remove(tmp_path)
                            os.rmdir(tmp_dir)
                        except OSError:
                            pass

                    except Exception as e:
                        st.error(f"Segmentation error: {e}")

        if st.session_state.interactive_result:
            res = st.session_state.interactive_result
            render_results(
                res["image_pil"], res["image_np"], res["masks"],
                caption="Point Segmentation Result", prefix="point"
            )

    # ── TEXT PROMPT MODE ──
    elif mode == "Text Prompt":
        st.subheader("📝 Text Prompt Segmentation")
        st.caption(
            "Describe what you want to segment. SAM 3 will find **all instances** "
            "of the concept in the image."
        )

        col_input, col_preview = st.columns([1, 2])

        with col_input:
            text_input = st.text_input(
                "What do you want to segment?",
                placeholder="e.g. person, red car, coffee mug",
            )

            multi_query = st.checkbox(
                "Multiple concepts (comma-separated)",
                value=False,
                help='e.g. "person, bicycle, dog" will segment each concept.',
            )

            run_btn = st.button("🔍 Segment", type="primary", use_container_width=True)

        with col_preview:
            st.image(image_pil, caption="Uploaded Image", use_container_width=True)

        if "text_result" not in st.session_state:
            st.session_state.text_result = None

        if run_btn and text_input.strip():
            if multi_query:
                queries = [q.strip() for q in text_input.split(",") if q.strip()]
            else:
                queries = [text_input.strip()]

            with st.spinner(f"Segmenting: {', '.join(queries)}..."):
                try:
                    tmp_path, tmp_dir = save_image_to_temp(image_pil)

                    predictor = load_semantic_predictor(model_path, conf=confidence)
                    predictor.args.conf = confidence

                    predictor.set_image(tmp_path)
                    src_shape = cv2.imread(tmp_path).shape[:2]

                    masks, boxes = predictor.inference_features(
                        predictor.features,
                        src_shape=src_shape,
                        text=queries,
                    )

                    masks_np = extract_masks_from_semantic(masks, boxes)

                    if masks_np is not None and len(masks_np) > 0:
                        st.session_state.text_result = {
                            "image_pil": image_pil,
                            "image_np": image_np,
                            "masks": masks_np,
                            "prompt": text_input,
                        }
                    else:
                        st.session_state.text_result = None
                        st.warning(
                            "No segments found. Try lowering the confidence "
                            "or using a different description."
                        )

                    try:
                        os.remove(tmp_path)
                        os.rmdir(tmp_dir)
                    except OSError:
                        pass

                except Exception as e:
                    st.error(f"Segmentation error: {e}")

        if st.session_state.text_result:
            res = st.session_state.text_result
            n = len(res["masks"])
            st.success(f"Found **{n}** segment(s)!")
            render_results(
                res["image_pil"], res["image_np"], res["masks"],
                caption=f"Result: {res['prompt']} ({n} matches)", prefix="text"
            )

    # Footer
    st.divider()
    st.caption(
        "Developed by [Zvonko Vugreshek](https://zvonkovugreshek.com) | [Pixolid UG](https://pixolid.de) · "
        "Powered by [SAM 3](https://docs.ultralytics.com/models/sam-3/) via Ultralytics · "
        "[Paper](https://arxiv.org/abs/2506.09832)"
    )


if __name__ == "__main__":
    main()
