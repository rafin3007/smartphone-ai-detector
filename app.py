
from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
import timm
import torch
from torch import nn
from torchvision import transforms
from damage_detection import (
    analyze_phone_damage,
    get_crack_overlay,
    get_scratch_overlay,
)


# ============================================================
# Application configuration
# ============================================================

st.set_page_config(
    page_title="Smartphone AI Detector",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
from huggingface_hub import hf_hub_download
import streamlit as st

HF_REPO = "mrafin/smartphone-classifier"
HF_FILE = "best_model.pth"

@st.cache_resource
def get_model_path():
    return hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILE,
    )

DEFAULT_MODEL_PATH = get_model_path()
DEFAULT_COMPONENTS_PATH = BASE_DIR / "data" / "smartphone_components_summary.csv"
DEFAULT_SPECS_PATH = BASE_DIR / "data" / "phone_specifications.json"

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
        ]
    }
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #071426 0%, #0c213c 55%, #102e50 100%);
    }

    [data-testid="stSidebar"] {
        background: #071426;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .hero {
        padding: 2.2rem;
        border-radius: 24px;
        background:
            radial-gradient(circle at top right, rgba(51,153,255,0.28), transparent 35%),
            linear-gradient(135deg, rgba(18,49,83,0.95), rgba(6,19,35,0.95));
        border: 1px solid rgba(125,195,255,0.20);
        box-shadow: 0 18px 50px rgba(0,0,0,0.22);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: clamp(2.2rem, 5vw, 4.2rem);
        margin: 0;
        color: #f8fbff;
        line-height: 1.05;
    }

    .hero p {
        color: #b8cee5;
        font-size: 1.1rem;
        max-width: 760px;
        margin-top: 1rem;
    }

    .prediction-card {
        border-radius: 20px;
        padding: 1.5rem;
        background: rgba(8, 27, 49, 0.88);
        border: 1px solid rgba(111,190,255,0.18);
        box-shadow: 0 12px 35px rgba(0,0,0,0.18);
        margin-top: 1rem;
    }

    .prediction-label {
        color: #8db7dc;
        text-transform: uppercase;
        letter-spacing: .12em;
        font-size: .76rem;
        margin-bottom: .35rem;
    }

    .prediction-name {
        color: #ffffff;
        font-weight: 750;
        font-size: 2rem;
        line-height: 1.15;
    }

    .confidence {
        color: #55d6be;
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: .55rem;
    }

    .small-note {
        color: #9eb6cc;
        font-size: .9rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(8, 27, 49, 0.88);
        border: 1px solid rgba(111,190,255,0.15);
        padding: 1rem;
        border-radius: 16px;
    }

    .status-ok {
        color: #54d6b9;
        font-weight: 700;
    }

    .status-bad {
        color: #ff8282;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Model loading and preprocessing
# ============================================================

def normalize_model_key(value: str) -> str:
    return "_".join(
        part for part in "".join(
            char.lower() if char.isalnum() else " "
            for char in value
        ).split()
        if part
    )


def pretty_class_name(value: str) -> str:
    special_tokens = {
        "iphone": "iPhone",
        "s24": "S24",
        "s25": "S25",
        "ultra": "Ultra",
        "pro": "Pro",
        "max": "Max",
        "edge": "Edge",
    }

    words = value.replace("-", "_").split("_")
    return " ".join(
        special_tokens.get(word.lower(), word.capitalize())
        for word in words
        if word
    )


@st.cache_resource(show_spinner="Loading classification model...")
def load_classifier(model_path: str) -> dict[str, Any]:
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint was not found: {path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model_name = checkpoint.get(
        "model_name",
        "convnextv2_tiny.fcmae_ft_in22k_in1k",
    )
    class_names = checkpoint.get("class_names")
    class_to_idx = checkpoint.get("class_to_idx")

    if class_names is None and class_to_idx:
        class_names = [
            name
            for name, _ in sorted(
                class_to_idx.items(),
                key=lambda item: item[1],
            )
        ]

    if not class_names:
        labels_path = path.parent.parent / "exports" / "class_names.json"

        if labels_path.exists():
            class_names = json.loads(
                labels_path.read_text(encoding="utf-8")
            )

    if not class_names:
        raise ValueError(
            "The checkpoint does not contain class_names or class_to_idx."
        )

    image_size = int(checkpoint.get("image_size", 320))
    mean = tuple(
        checkpoint.get(
            "mean",
            (0.485, 0.456, 0.406),
        )
    )
    std = tuple(
        checkpoint.get(
            "std",
            (0.229, 0.224, 0.225),
        )
    )

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=len(class_names),
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint),
    )

    # Remove prefixes added by DataParallel or some training wrappers.
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        cleaned_key = key

        for prefix in ("module.", "model."):
            if cleaned_key.startswith(prefix):
                cleaned_key = cleaned_key[len(prefix):]

        cleaned_state_dict[cleaned_key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    # A classifier mismatch usually means the wrong checkpoint was selected.
    serious_missing = [
        key for key in missing
        if "head" not in key and "classifier" not in key and "fc" not in key
    ]

    if serious_missing:
        raise RuntimeError(
            "Checkpoint and architecture do not match. "
            f"Missing keys include: {serious_missing[:8]}"
        )

    model.to(device)
    model.eval()

    preprocessing = transforms.Compose(
        [
            transforms.Resize(
                int(image_size * 1.10),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return {
        "model": model,
        "device": device,
        "class_names": class_names,
        "image_size": image_size,
        "mean": mean,
        "std": std,
        "preprocessing": preprocessing,
        "model_name": model_name,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def predict_pil_image(
    image: Image.Image,
    model_bundle: dict[str, Any],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    model: nn.Module = model_bundle["model"]
    device: torch.device = model_bundle["device"]
    preprocessing = model_bundle["preprocessing"]
    class_names = model_bundle["class_names"]

    rgb_image = image.convert("RGB")
    tensor = preprocessing(rgb_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

    top_k = min(top_k, len(class_names))
    values, indices = probabilities.topk(top_k)

    return [
        {
            "class_name": class_names[int(index)],
            "display_name": pretty_class_name(
                class_names[int(index)]
            ),
            "confidence": float(value),
        }
        for value, index in zip(
            values.detach().cpu(),
            indices.detach().cpu(),
        )
    ]


# ============================================================
# Phone specification lookup
# ============================================================

@st.cache_data(show_spinner=False)
def load_phone_specifications(path: str) -> dict[str, Any]:
    specification_path = Path(path)
    if not specification_path.exists():
        return {}
    payload = json.loads(specification_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("phone_specifications.json must contain a JSON object.")
    return {normalize_model_key(str(key)): value for key, value in payload.items() if isinstance(value, dict)}

def lookup_phone_specification(predicted_class: str, specifications: dict[str, Any]) -> dict[str, Any] | None:
    key = normalize_model_key(predicted_class)
    if key in specifications:
        return specifications[key]
    for candidate, value in specifications.items():
        if candidate in key or key in candidate:
            return value
    return None

def render_phone_specification(predicted_class: str, specifications: dict[str, Any]) -> None:
    spec = lookup_phone_specification(predicted_class, specifications)
    if not spec:
        st.info("No specification record was found for this predicted model.")
        return
    st.subheader("Phone specifications")
    display = spec.get("display") or {}
    battery = spec.get("battery") or {}
    metrics = st.columns(4)
    metrics[0].metric("Brand", spec.get("brand", "—"))
    metrics[1].metric("Model", spec.get("model", pretty_class_name(predicted_class)))
    metrics[2].metric("Release year", spec.get("release_year", "—"))
    weight = spec.get("weight_g")
    metrics[3].metric("Weight", f"{weight} g" if weight else "—")
    rows = [
        ("Operating system", spec.get("operating_system", "—")),
        ("Chipset", spec.get("chipset", "—")),
        ("RAM", f"{spec.get('ram_gb')} GB" if spec.get('ram_gb') else "—"),
        ("Storage", ", ".join(f"{v} GB" for v in spec.get("storage_options_gb", [])) or "—"),
        ("Display", f"{display.get('size_inches', '—')} in {display.get('type', '')}".strip()),
        ("Resolution", display.get("resolution", "—")),
        ("Refresh rate", f"{display.get('refresh_rate_hz')} Hz" if display.get("refresh_rate_hz") else "—"),
        ("Rear camera", spec.get("rear_camera", "—")),
        ("Front camera", spec.get("front_camera", "—")),
        ("Battery", f"{battery.get('capacity_mah')} mAh" if battery.get("capacity_mah") else "—"),
        ("Charging", battery.get("charging", "—")),
        ("Dimensions", spec.get("dimensions_mm", "—")),
        ("Network", spec.get("network", "—")),
        ("Water resistance", spec.get("water_resistance", "—")),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Specification", "Value"]), use_container_width=True, hide_index=True)

# ============================================================
# Optional component lookup
# ============================================================

@st.cache_data(show_spinner=False)
def load_component_dataset(path: str) -> pd.DataFrame:
    component_path = Path(path)

    if not component_path.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(component_path)

    candidate_columns = [
        "requested_model",
        "model",
        "model_key",
    ]

    source_column = next(
        (
            column for column in candidate_columns
            if column in dataframe.columns
        ),
        None,
    )

    if source_column:
        dataframe["_model_key"] = (
            dataframe[source_column]
            .astype(str)
            .map(normalize_model_key)
        )

    return dataframe


def lookup_components(
    predicted_class: str,
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    if dataframe.empty or "_model_key" not in dataframe.columns:
        return pd.DataFrame()

    model_key = normalize_model_key(predicted_class)
    matches = dataframe[
        dataframe["_model_key"] == model_key
    ].copy()

    if matches.empty:
        # Conservative partial fallback, useful when one source includes a brand.
        matches = dataframe[
            dataframe["_model_key"].str.contains(
                model_key,
                regex=False,
                na=False,
            )
            | pd.Series(
                [
                    key in model_key
                    for key in dataframe["_model_key"]
                ],
                index=dataframe.index,
            )
        ].copy()

    return matches


# ============================================================
# Damage rendering
# ============================================================

def render_damage_analysis(image: Image.Image) -> None:
    st.subheader("Visible damage analysis")
    st.caption(
        "This estimates visible surface damage from the submitted photo. "
        "Reflections, blur, cases, screen content, and lighting can affect the result."
    )

    with st.spinner("Analyzing cracks, scratches, and visible damage..."):
        damage_result, scratch_mask, crack_mask = analyze_phone_damage(image)

    quality = damage_result.get("image_quality", {})
    warnings = quality.get("warnings", [])
    if warnings:
        st.warning("Image-quality warning: " + "; ".join(warnings))

    confidence = damage_result.get("damage_confidence", {})
    cards = st.columns(5)
    damage_items = [
        ("Cracked screen", damage_result.get("cracked_screen", False), confidence.get("crack", 0.0)),
        ("Broken display", damage_result.get("broken_display", False), confidence.get("broken_display", 0.0)),
        ("Scratches", damage_result.get("visible_scratches", False), confidence.get("scratch", 0.0)),
        ("Body / back", damage_result.get("damaged_back_or_body", False), confidence.get("body", 0.0)),
        ("Camera lens", damage_result.get("camera_lens_damage", False), confidence.get("camera_lens", 0.0)),
    ]

    for column, (label, detected, score) in zip(cards, damage_items):
        with column:
            st.metric(label, "Detected" if detected else "Not detected", f"{score * 100:.1f}% confidence")

    summary_columns = st.columns(3)
    summary_columns[0].metric("Overall condition", damage_result.get("image_condition", "Unknown"))
    summary_columns[1].metric("Scratch severity", damage_result.get("scratch_severity", "None"))
    summary_columns[2].metric("Image quality", f"{quality.get('quality_score', 0.0) * 100:.0f}%")

    overlay_tabs = st.tabs(["Scratch overlay", "Crack overlay", "Technical details"])
    with overlay_tabs[0]:
        if damage_result.get("visible_scratches", False):
            st.image(get_scratch_overlay(image, scratch_mask), caption="Potential scratches highlighted in yellow/orange.", use_container_width=True)
        else:
            st.success("No strong scratch pattern was detected.")
    with overlay_tabs[1]:
        if damage_result.get("cracked_screen", False):
            st.image(get_crack_overlay(image, crack_mask), caption="Potential cracks highlighted in red.", use_container_width=True)
        else:
            st.success("No strong crack pattern was detected.")
    with overlay_tabs[2]:
        st.json(damage_result)

# ============================================================
# Result rendering
# ============================================================

def render_prediction(
    predictions: list[dict[str, Any]],
    component_dataframe: pd.DataFrame,
    specifications: dict[str, Any],
    threshold: float,
    image: Image.Image | None = None,
) -> None:
    if not predictions:
        return

    top = predictions[0]

    if top["confidence"] < threshold:
        st.warning(
            "The prediction is below the selected confidence threshold. "
            "Try a clearer rear-camera or full-device view."
        )

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-label">Predicted smartphone</div>
            <div class="prediction-name">{top["display_name"]}</div>
            <div class="confidence">
                Confidence: {top["confidence"] * 100:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Top predictions")

    chart_dataframe = pd.DataFrame(
        {
            "Model": [
                item["display_name"]
                for item in predictions
            ],
            "Confidence": [
                item["confidence"]
                for item in predictions
            ],
        }
    ).set_index("Model")

    st.bar_chart(chart_dataframe)

    if image is not None:
        render_damage_analysis(image)

    render_phone_specification(top["class_name"], specifications)

    matches = lookup_components(
        top["class_name"],
        component_dataframe,
    )

    if matches.empty:
        st.info(
            "No matching component records were found. "
            "The smartphone prediction still works without the component CSV."
        )
        return

    st.subheader("Expected internal components")
    st.caption(
        "These components come from the component database for the predicted "
        "model. They are not visually detected through the exterior."
    )

    board_column = next(
        (
            column
            for column in ("is_pcb_or_board", "pcb_type")
            if column in matches.columns
        ),
        None,
    )

    if board_column == "is_pcb_or_board":
        board_mask = (
            matches[board_column]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
    elif board_column == "pcb_type":
        board_mask = matches[board_column].notna()
    else:
        board_mask = pd.Series(False, index=matches.index)

    boards = matches[board_mask]
    other_components = matches[~board_mask]

    first, second = st.columns(2)

    component_name_column = next(
        (
            column
            for column in ("component_name", "component", "pcb_type")
            if column in matches.columns
        ),
        None,
    )

    with first:
        st.markdown("#### PCB and boards")

        if boards.empty or component_name_column is None:
            st.write("No PCB records available.")
        else:
            for value in boards[component_name_column].dropna().unique():
                st.write(f"• {value}")

    with second:
        st.markdown("#### Other components")

        if other_components.empty or component_name_column is None:
            st.write("No component records available.")
        else:
            for value in (
                other_components[component_name_column]
                .dropna()
                .unique()[:30]
            ):
                st.write(f"• {value}")


# ============================================================
# Live WebRTC processor
# ============================================================

class SmartphoneVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model_bundle: dict[str, Any] | None = None
        self.threshold = 0.60
        self.frame_interval = 10
        self.frame_count = 0
        self.last_label = "Point the camera at a smartphone"
        self.last_confidence = 0.0
        self.last_inference_ms = 0.0

    def configure(
        self,
        model_bundle: dict[str, Any],
        threshold: float,
        frame_interval: int,
    ) -> None:
        with self.lock:
            self.model_bundle = model_bundle
            self.threshold = threshold
            self.frame_interval = max(1, frame_interval)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        with self.lock:
            bundle = self.model_bundle
            threshold = self.threshold
            frame_interval = self.frame_interval

        if bundle is not None and self.frame_count % frame_interval == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)

            started = time.perf_counter()

            try:
                prediction = predict_pil_image(
                    image,
                    bundle,
                    top_k=1,
                )[0]

                elapsed_ms = (
                    time.perf_counter() - started
                ) * 1000

                self.last_label = prediction["display_name"]
                self.last_confidence = prediction["confidence"]
                self.last_inference_ms = elapsed_ms

            except Exception as error:
                self.last_label = f"Inference error: {type(error).__name__}"
                self.last_confidence = 0.0

        overlay = bgr.copy()

        # Dark translucent prediction panel.
        cv2.rectangle(
            overlay,
            (18, 18),
            (min(bgr.shape[1] - 18, 640), 126),
            (7, 20, 36),
            thickness=-1,
        )

        bgr = cv2.addWeighted(
            overlay,
            0.78,
            bgr,
            0.22,
            0,
        )

        confident = self.last_confidence >= threshold
        status_text = (
            self.last_label
            if confident
            else f"Uncertain: {self.last_label}"
        )

        cv2.putText(
            bgr,
            status_text,
            (36, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.86,
            (90, 230, 190) if confident else (90, 190, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            bgr,
            f"Confidence: {self.last_confidence * 100:.1f}%  "
            f"|  Inference: {self.last_inference_ms:.0f} ms",
            (36, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 232, 243),
            1,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(
            bgr,
            format="bgr24",
        )


# ============================================================
# User interface
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Smartphone AI Detector</h1>
        <p>
            Identify a smartphone model from an uploaded photo, a camera
            snapshot, or a live webcam feed. The classifier can also connect
            the predicted model to your PCB and internal-component dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model configuration")

    model_path = st.text_input(
        "Checkpoint path",
        value=str(DEFAULT_MODEL_PATH),
        help=(
            "Use your best_model.pth checkpoint. When deploying, place it "
            "inside the app's models folder."
        ),
    )

    components_path = st.text_input(
        "Component dataset path",
        value=str(DEFAULT_COMPONENTS_PATH),
        help="Optional CSV produced by the component-dataset pipeline.",
    )

    specifications_path = st.text_input(
        "Phone specifications JSON path",
        value=str(DEFAULT_SPECS_PATH),
        help="JSON containing model-specific phone specifications.",
    )

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.30,
        max_value=0.95,
        value=0.60,
        step=0.05,
    )

    top_k = st.slider(
        "Number of predictions",
        min_value=1,
        max_value=5,
        value=3,
    )

    live_frame_interval = st.slider(
        "Live inference interval",
        min_value=1,
        max_value=30,
        value=10,
        help=(
            "Run the model once every N video frames. A higher value reduces "
            "CPU/GPU usage."
        ),
    )

    st.divider()
    st.caption(
        "For the clearest result, show the complete phone or the rear camera "
        "layout without a case."
    )

try:
    model_bundle = load_classifier(model_path)
    model_ready = True
    model_error = None
except Exception as error:
    model_bundle = None
    model_ready = False
    model_error = error

component_dataframe = load_component_dataset(components_path)
phone_specifications = load_phone_specifications(specifications_path)

status_columns = st.columns(5)

with status_columns[0]:
    st.metric(
        "Model status",
        "Ready" if model_ready else "Not loaded",
    )

with status_columns[1]:
    st.metric(
        "Runtime",
        (
            str(model_bundle["device"]).upper()
            if model_ready
            else "—"
        ),
    )

with status_columns[2]:
    st.metric(
        "Classes",
        (
            len(model_bundle["class_names"])
            if model_ready
            else 0
        ),
    )

with status_columns[3]:
    st.metric(
        "Component records",
        len(component_dataframe),
    )

with status_columns[4]:
    st.metric(
        "Specification records",
        len(phone_specifications),
    )

if not model_ready:
    st.error(
        f"Could not load the model: {model_error}\n\n"
        "Copy your best_model.pth into models/best_model.pth or update the "
        "checkpoint path in the sidebar."
    )
    st.stop()

mode = st.tabs(
    [
        "Upload image",
        "Camera snapshot",
        "Live camera",
        "Model information",
    ]
)

with mode[0]:
    first, second = st.columns(
        [1.05, 0.95],
        gap="large",
    )

    with first:
        uploaded_file = st.file_uploader(
            "Upload a smartphone image",
            type=["jpg", "jpeg", "png", "webp"],
        )

        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            st.image(
                uploaded_image,
                caption="Uploaded image",
                use_container_width=True,
            )

    with second:
        if uploaded_file is not None:
            with st.spinner("Analyzing smartphone..."):
                predictions = predict_pil_image(
                    uploaded_image,
                    model_bundle,
                    top_k=top_k,
                )

            render_prediction(
                predictions,
                component_dataframe,
                phone_specifications,
                confidence_threshold,
                image=uploaded_image,
            )
        else:
            st.info("Upload an image to begin.")

with mode[1]:
    snapshot_columns = st.columns(
        [1.05, 0.95],
        gap="large",
    )

    with snapshot_columns[0]:
        camera_file = st.camera_input(
            "Take a clear smartphone photo"
        )

    with snapshot_columns[1]:
        if camera_file is not None:
            camera_image = Image.open(camera_file).convert("RGB")

            with st.spinner("Analyzing camera image..."):
                predictions = predict_pil_image(
                    camera_image,
                    model_bundle,
                    top_k=top_k,
                )

            render_prediction(
                predictions,
                component_dataframe,
                phone_specifications,
                confidence_threshold,
                image=camera_image,
            )
        else:
            st.info("Allow camera access and take a picture.")

with mode[2]:
    st.subheader("Real-time smartphone recognition")
    st.caption(
        "Press START, allow browser camera access, and point the camera at "
        "the phone. The prediction is drawn on the video."
    )

    context = webrtc_streamer(
        key="smartphone-live-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=SmartphoneVideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 960},
                "height": {"ideal": 720},
                "facingMode": "environment",
            },
            "audio": False,
        },
        async_processing=True,
    )

    if context.video_processor:
        context.video_processor.configure(
            model_bundle=model_bundle,
            threshold=confidence_threshold,
            frame_interval=live_frame_interval,
        )

    st.info(
        "Live classification is frame-based. A lower inference interval is "
        "more responsive but uses more computing resources."
    )

with mode[3]:
    st.subheader("Loaded model")

    model_details = {
        "Architecture": model_bundle["model_name"],
        "Device": str(model_bundle["device"]),
        "Input size": model_bundle["image_size"],
        "Classes": len(model_bundle["class_names"]),
        "Normalization mean": model_bundle["mean"],
        "Normalization standard deviation": model_bundle["std"],
    }

    st.json(model_details)

    st.subheader("Recognized classes")

    class_table = pd.DataFrame(
        {
            "Training label": model_bundle["class_names"],
            "Display name": [
                pretty_class_name(value)
                for value in model_bundle["class_names"]
            ],
        }
    )

    st.dataframe(
        class_table,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "The model can recognize only the classes present during training. "
        "Unknown phone models may be forced into the closest known class, so "
        "use the confidence threshold and add an unknown-device class for a "
        "production system."
    )
