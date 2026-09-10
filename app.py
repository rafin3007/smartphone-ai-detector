
from __future__ import annotations

import io
import re
import json
import threading
import time
from pathlib import Path
from difflib import SequenceMatcher
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


@st.cache_resource(show_spinner="Downloading model from Hugging Face...")
def get_model_path():
    return hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILE,
        repo_type="model",
    )


DEFAULT_MODEL_PATH = ""
DEFAULT_COMPONENTS_PATH = BASE_DIR / "data" / "smartphone_components_summary.csv"
DEFAULT_SPECS_PATH = BASE_DIR / "data" / "phone_specifications.json"
DEFAULT_METALS_PATH = BASE_DIR / "data" / "valuable_metals.json"
DEFAULT_COMPONENT_COUNTS_PATH = BASE_DIR / "data" / "smartphone_component_counts.json"
DEFAULT_PRICING_PATH = BASE_DIR / "data" / "early_upgrade_pricing.xlsx"

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
    /* =====================================================
       GLOBAL WHITE / LIGHT-GRAY THEME
       ===================================================== */

    html, body {
        color-scheme: light !important;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: #ffffff !important;
        color: #111111 !important;
    }

    [data-testid="stHeader"] {
        background: #ffffff !important;
        border-bottom: 1px solid #e5e7eb !important;
    }

    [data-testid="stSidebar"] {
        background: #f7f7f8 !important;
        border-right: 1px solid #e5e7eb !important;
    }

    [data-testid="stSidebar"] * {
        color: #111111 !important;
    }

    /* =====================================================
       HERO
       ===================================================== */

    .hero {
        padding: 2.2rem;
        border-radius: 24px;
        background: #f3f4f6 !important;
        border: 1px solid #d1d5db;
        box-shadow: 0 10px 28px rgba(0,0,0,0.07);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: clamp(2.2rem, 5vw, 4.2rem);
        margin: 0;
        color: #111111 !important;
        line-height: 1.05;
    }

    .hero p {
        color: #333333 !important;
        font-size: 1.1rem;
        max-width: 760px;
        margin-top: 1rem;
    }

    /* =====================================================
       RESULT / VALUATION CARDS
       ===================================================== */

    .prediction-card {
        border-radius: 20px;
        padding: 1.5rem;
        background: #f3f4f6 !important;
        border: 1px solid #d1d5db;
        box-shadow: 0 8px 22px rgba(0,0,0,0.06);
        margin-top: 1rem;
    }

    .prediction-label {
        color: #444444 !important;
        text-transform: uppercase;
        letter-spacing: .12em;
        font-size: .76rem;
        margin-bottom: .35rem;
    }

    .prediction-name {
        color: #111111 !important;
        font-weight: 750;
        font-size: 2rem;
        line-height: 1.15;
    }

    .confidence {
        color: #111111 !important;
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: .55rem;
    }

    .small-note {
        color: #4b5563 !important;
        font-size: .9rem;
    }

    /* =====================================================
       GENERAL TEXT
       ===================================================== */

    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown span,
    .stCaption,
    label,
    h1, h2, h3, h4, h5, h6,
    [data-testid="stWidgetLabel"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        color: #111111 !important;
    }

    /* Metric value */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        line-height: 1.1 !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }

    /* Metric label */
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
    }

    /* =====================================================
       TABS
       ===================================================== */

    [data-baseweb="tab-list"] {
        background: #ffffff !important;
        border-bottom: 1px solid #e5e7eb !important;
    }

    [data-baseweb="tab"] {
        color: #111111 !important;
        background: transparent !important;
    }

    [aria-selected="true"][data-baseweb="tab"] {
        color: #111111 !important;
        font-weight: 700 !important;
    }

    /* =====================================================
       CHAT
       ===================================================== */

    [data-testid="stChatMessage"] {
        background: #f8f9fa !important;
        border: 1px solid #e5e7eb !important;
        color: #111111 !important;
        border-radius: 16px;
        padding: 0.45rem 0.65rem;
        margin-bottom: 0.45rem;
    }

    [data-testid="stChatMessage"] * {
        color: #111111 !important;
    }

    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 18px !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #111111 !important;
        background: #ffffff !important;
        font-size: 1rem !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #6b7280 !important;
    }

    /* =====================================================
       INPUTS / SELECTS
       ===================================================== */

    [data-baseweb="input"] > div,
    [data-baseweb="select"] > div,
    [data-baseweb="textarea"] > div {
        background: #ffffff !important;
        color: #111111 !important;
        border-color: #d1d5db !important;
    }

    input,
    textarea,
    select {
        color: #111111 !important;
        background: #ffffff !important;
    }

    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button,
    .stDownloadButton > button {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #f3f4f6 !important;
        color: #000000 !important;
        border-color: #94a3b8 !important;
    }

    .stButton > button *,
    .stDownloadButton > button * {
        color: #000000 !important;
    }

    /* =====================================================
       BORDERED CONTAINERS / EXPANDERS
       ===================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border-color: #d1d5db !important;
        border-radius: 18px;
    }

    [data-testid="stExpander"] {
        background: #f8f9fa !important;
        border: 1px solid #e5e7eb !important;
    }

    [data-testid="stExpander"] * {
        color: #111111 !important;
    }

    /* =====================================================
       DATAFRAMES
       ===================================================== */

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background: #ffffff !important;
        color: #111111 !important;
    }

    /* =====================================================
       PROGRESS
       ===================================================== */

    [data-testid="stProgress"] > div > div {
        background: #d1d5db !important;
    }

    [data-testid="stProgress"] > div > div > div {
        background: #4b5563 !important;
    }

    /* =====================================================
       STATUS COLORS
       ===================================================== */

    .status-ok {
        color: #166534 !important;
        font-weight: 700;
    }

    .status-bad {
        color: #b91c1c !important;
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
# Valuable metals lookup
# ============================================================

@st.cache_data(show_spinner=False)
def load_valuable_metals(path: str) -> dict[str, Any]:
    metals_path = Path(path)

    if not metals_path.exists():
        return {}

    with metals_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            "valuable_metals.json must contain a JSON object."
        )

    payload.pop("metadata", None)

    return {
        normalize_model_key(str(model_key)): model_data
        for model_key, model_data in payload.items()
        if isinstance(model_data, dict)
    }


def lookup_valuable_metals(
    predicted_class: str,
    metals_data: dict[str, Any],
) -> dict[str, Any] | None:
    model_key = normalize_model_key(predicted_class)

    if model_key in metals_data:
        return metals_data[model_key]

    for key, value in metals_data.items():
        if key in model_key or model_key in key:
            return value

    return None


def _format_metal_quantity(quantity_g: float) -> str:
    if quantity_g < 0.1:
        return f"{quantity_g * 1000:.1f} mg"
    return f"{quantity_g:.2f} g"


def _format_metal_range(
    minimum: float | None,
    maximum: float | None,
) -> str:
    if minimum is None or maximum is None:
        return "—"

    if maximum < 0.1:
        return f"{minimum * 1000:.1f}–{maximum * 1000:.1f} mg"

    return f"{minimum:.3f}–{maximum:.3f} g"


def render_valuable_metals(
    predicted_class: str,
    metals_data: dict[str, Any],
) -> None:
    phone_metals = lookup_valuable_metals(
        predicted_class,
        metals_data,
    )

    if not phone_metals:
        st.info(
            "Estimated valuable-metal information is not available "
            "for this model."
        )
        return

    st.subheader("Estimated valuable-metal content")
    st.caption(
        "These values are research-based estimates, not exact "
        "manufacturer-reported measurements. Actual quantities can vary "
        "by production revision, supplier, region, and configuration."
    )

    brand = phone_metals.get("brand", "—")
    model_name = phone_metals.get(
        "model",
        pretty_class_name(predicted_class),
    )
    confidence = (
        str(phone_metals.get("confidence", "unknown"))
        .replace("_", " ")
        .title()
    )
    weight = phone_metals.get("device_weight_g", "—")

    overview_columns = st.columns(4)
    overview_columns[0].metric("Brand", brand)
    overview_columns[1].metric("Model", model_name)
    overview_columns[2].metric("Estimate confidence", confidence)
    overview_columns[3].metric(
        "Device weight",
        f"{weight} g" if weight != "—" else "—",
    )

    metals = phone_metals.get("valuable_metals", {})
    if not metals:
        st.info("No metal entries are available for this model.")
        return

    rows = []

    for metal_name, metal_info in metals.items():
        if not isinstance(metal_info, dict):
            continue

        quantity_g = float(
            metal_info.get("estimated_quantity_g", 0)
        )
        quantity_range = metal_info.get(
            "estimated_range_g",
            {},
        ) or {}

        rows.append(
            {
                "Metal": metal_name.replace("_", " ").title(),
                "Symbol": metal_info.get("symbol", ""),
                "Estimated quantity": _format_metal_quantity(quantity_g),
                "Estimated range": _format_metal_range(
                    quantity_range.get("minimum"),
                    quantity_range.get("maximum"),
                ),
                "Main locations": ", ".join(
                    metal_info.get("primary_locations", [])
                ) or "—",
                "quantity_g": quantity_g,
            }
        )

    if not rows:
        st.info("No metal entries are available for this model.")
        return

    metals_dataframe = pd.DataFrame(rows)

    for start_index in range(0, min(len(rows), 8), 4):
        columns = st.columns(4)
        for column, item in zip(
            columns,
            rows[start_index:start_index + 4],
        ):
            with column:
                st.metric(
                    label=f"{item['Metal']} ({item['Symbol']})",
                    value=item["Estimated quantity"],
                )

    st.markdown("#### Complete metal details")
    st.dataframe(
        metals_dataframe[
            [
                "Metal",
                "Symbol",
                "Estimated quantity",
                "Estimated range",
                "Main locations",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    chart_dataframe = (
        metals_dataframe[["Metal", "quantity_g"]]
        .set_index("Metal")
        .rename(columns={"quantity_g": "Estimated quantity (g)"})
    )

    st.markdown("#### Estimated quantity comparison")
    st.bar_chart(chart_dataframe)

    with st.expander("Method and interpretation"):
        st.write(
            "Contained quantity is not the same as recoverable quantity. "
            "Actual recycling recovery depends on collection, dismantling, "
            "separation, refining technology, and process efficiency."
        )


# ============================================================
# Component quantity and part-number lookup
# ============================================================

@st.cache_data(show_spinner=False)
def load_component_counts(path: str) -> dict[str, Any]:
    component_path = Path(path)

    if not component_path.exists():
        return {}

    with component_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            "smartphone_component_counts.json must contain a JSON object."
        )

    return {
        normalize_model_key(str(model_key)): model_data
        for model_key, model_data in payload.items()
        if isinstance(model_data, dict)
    }


def lookup_component_counts(
    predicted_class: str,
    component_counts: dict[str, Any],
) -> dict[str, Any] | None:
    model_key = normalize_model_key(predicted_class)

    if model_key in component_counts:
        return component_counts[model_key]

    for key, value in component_counts.items():
        if key in model_key or model_key in key:
            return value

    return None


def render_component_counts(
    predicted_class: str,
    component_counts: dict[str, Any],
) -> None:
    phone_data = lookup_component_counts(
        predicted_class,
        component_counts,
    )

    if not phone_data:
        st.info(
            "Component quantity and part-number information is not available "
            "for this smartphone."
        )
        return

    st.subheader("Internal components, quantity, and part numbers")
    st.caption(
        "Part numbers may include internal dataset IDs and, where verified, "
        "manufacturer/service identifiers. A blank manufacturer part number "
        "means it has not yet been verified."
    )

    components = phone_data.get("components", [])
    if not isinstance(components, list) or not components:
        st.info("No component-count records are available for this model.")
        return

    rows = []

    for component in components:
        if not isinstance(component, dict):
            continue

        quantity = component.get("quantity", 1)
        try:
            quantity = int(quantity)
        except (TypeError, ValueError):
            quantity = 1

        rows.append(
            {
                "Part Number": component.get("part_number", "—"),
                "Manufacturer Part Number": (
                    component.get("manufacturer_part_number") or "—"
                ),
                "Component": component.get("component_name", "Unknown"),
                "Quantity": quantity,
                "Subcomponents": (
                    component.get("subcomponent_count")
                    if component.get("subcomponent_count") is not None
                    else "—"
                ),
                "Category": component.get("category", "—"),
                "Location": component.get("location", "—"),
                "Function": component.get("function", "—"),
            }
        )

    if not rows:
        st.info("No component-count records are available for this model.")
        return

    component_df = pd.DataFrame(rows)
    total_units = int(
        pd.to_numeric(component_df["Quantity"], errors="coerce")
        .fillna(0)
        .sum()
    )
    verified_part_numbers = int(
        (component_df["Manufacturer Part Number"] != "—").sum()
    )

    summary_columns = st.columns(4)
    summary_columns[0].metric(
        "Model",
        phone_data.get("model", pretty_class_name(predicted_class)),
    )
    summary_columns[1].metric("Component types", len(component_df))
    summary_columns[2].metric("Total units", total_units)
    summary_columns[3].metric(
        "Verified part numbers",
        verified_part_numbers,
    )

    st.dataframe(
        component_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Quantity": st.column_config.NumberColumn(
                "Qty",
                format="%d",
            ),
        },
    )

    category_summary = (
        component_df.groupby("Category", dropna=False)["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .to_frame("Units")
    )

    if not category_summary.empty:
        st.markdown("#### Component units by category")
        st.bar_chart(category_summary)

    with st.expander("Part-number notes"):
        st.write(
            "The Part Number column can contain your stable research dataset "
            "identifier, such as APL-IP15P-PCB-001. The Manufacturer Part "
            "Number column should only be populated when the identifier has "
            "been verified from a reliable source."
        )


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
    component_counts: dict[str, Any],
    specifications: dict[str, Any],
    metals_data: dict[str, Any],
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

    render_phone_specification(
        top["class_name"],
        specifications,
    )

    render_component_counts(
        top["class_name"],
        component_counts,
    )

    render_valuable_metals(
        top["class_name"],
        metals_data,
    )

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
# Manual valuation assistant + Excel pricing
# ============================================================

@st.cache_data(show_spinner=False)
def load_pricing_workbook(path: str) -> dict[str, Any]:
    """Load and normalize the Early Upgrade pricing workbook.

    Sheet names are resolved case-insensitively and with surrounding
    whitespace ignored so that minor Excel/GitHub naming differences do
    not break the Streamlit app.
    """
    pricing_path = Path(path)

    empty_result = {
        "available": False,
        "error": None,
        "phones": pd.DataFrame(),
        "tablets": pd.DataFrame(),
        "terms": {},
        "categories": {},
        "sheet_names": [],
    }

    if not pricing_path.exists():
        empty_result["error"] = (
            f"Pricing workbook not found: {pricing_path}"
        )
        return empty_result

    try:
        excel_file = pd.ExcelFile(
            pricing_path,
            engine="openpyxl",
        )

        workbook_sheet_names = list(
            excel_file.sheet_names
        )

        def normalize_sheet_name(value: str) -> str:
            return " ".join(
                str(value).strip().lower().split()
            )

        normalized_sheets = {
            normalize_sheet_name(sheet_name): sheet_name
            for sheet_name in workbook_sheet_names
        }

        def resolve_sheet(
            preferred_names: list[str],
            fallback_index: int | None = None,
        ) -> str:
            for preferred in preferred_names:
                key = normalize_sheet_name(preferred)

                if key in normalized_sheets:
                    return normalized_sheets[key]

            # Flexible contains-match fallback.
            for preferred in preferred_names:
                preferred_key = normalize_sheet_name(
                    preferred
                )

                for normalized, original in (
                    normalized_sheets.items()
                ):
                    if (
                        preferred_key in normalized
                        or normalized in preferred_key
                    ):
                        return original

            if (
                fallback_index is not None
                and fallback_index < len(
                    workbook_sheet_names
                )
            ):
                return workbook_sheet_names[
                    fallback_index
                ]

            raise ValueError(
                "Could not find any of these worksheets: "
                f"{preferred_names}. Available worksheets: "
                f"{workbook_sheet_names}"
            )

        phones_sheet = resolve_sheet(
            ["phones", "phone", "smartphones"],
            fallback_index=0,
        )
        tablets_sheet = resolve_sheet(
            ["tablets", "tablet"],
            fallback_index=1,
        )
        terms_sheet = resolve_sheet(
            ["terms", "criteria", "conditions"],
            fallback_index=2,
        )
        categories_sheet = resolve_sheet(
            [
                "Categories wprice",
                "categories w price",
                "categories",
                "commodity",
            ],
            fallback_index=3,
        )

        phones_raw = pd.read_excel(
            excel_file,
            sheet_name=phones_sheet,
            header=None,
        )

        tablets_raw = pd.read_excel(
            excel_file,
            sheet_name=tablets_sheet,
            header=None,
        )

        terms_raw = pd.read_excel(
            excel_file,
            sheet_name=terms_sheet,
            header=None,
        )

        categories_raw = pd.read_excel(
            excel_file,
            sheet_name=categories_sheet,
            header=None,
        )

    except Exception as error:
        empty_result["error"] = (
            f"Could not read pricing workbook: {error}"
        )

        try:
            empty_result["sheet_names"] = (
                pd.ExcelFile(
                    pricing_path,
                    engine="openpyxl",
                ).sheet_names
            )
        except Exception:
            pass

        return empty_result

    phone_rows: list[dict[str, Any]] = []

    # Sheet layout: iPhone A:E and Samsung F:J.
    for _, row in phones_raw.iloc[1:].iterrows():
        for model_col, start_col, brand in (
            (0, 1, "Apple"),
            (5, 6, "Samsung"),
        ):
            model = row.iloc[model_col] if model_col < len(row) else None

            if pd.isna(model) or not str(model).strip():
                continue

            phone_rows.append(
                {
                    "model": str(model).strip(),
                    "model_key": normalize_model_key(str(model)),
                    "brand": brand,
                    "device_type": "phone",
                    "PTG": row.iloc[start_col] if start_col < len(row) else None,
                    "PTC": row.iloc[start_col + 1] if start_col + 1 < len(row) else None,
                    "PBL": row.iloc[start_col + 2] if start_col + 2 < len(row) else None,
                    "NP": row.iloc[start_col + 3] if start_col + 3 < len(row) else None,
                }
            )

    tablet_rows: list[dict[str, Any]] = []

    # Sheet layout: OFF A:E and ON F:J. The final ON header is blank,
    # but its position mirrors the NP column in the OFF section.
    for _, row in tablets_raw.iloc[1:].iterrows():
        model = row.iloc[0] if len(row) > 0 else None

        if pd.isna(model) or not str(model).strip():
            continue

        model_text = str(model).strip()

        tablet_rows.append(
            {
                "model": model_text,
                "model_key": normalize_model_key(model_text),
                "brand": "Apple" if "ipad" in model_text.lower() else "Tablet",
                "device_type": "tablet",
                "cloud_status": "off",
                "PTG": row.iloc[1] if len(row) > 1 else None,
                "PTC": row.iloc[2] if len(row) > 2 else None,
                "PBL": row.iloc[3] if len(row) > 3 else None,
                "NP": row.iloc[4] if len(row) > 4 else None,
            }
        )

        tablet_rows.append(
            {
                "model": model_text,
                "model_key": normalize_model_key(model_text),
                "brand": "Apple" if "ipad" in model_text.lower() else "Tablet",
                "device_type": "tablet",
                "cloud_status": "on",
                "PTG": row.iloc[6] if len(row) > 6 else None,
                "PTC": row.iloc[7] if len(row) > 7 else None,
                "PBL": row.iloc[8] if len(row) > 8 else None,
                "NP": row.iloc[9] if len(row) > 9 else None,
            }
        )

    terms: dict[str, str] = {}
    for value in terms_raw.iloc[:, 0].dropna().astype(str):
        cleaned = value.strip()
        upper = cleaned.upper()

        for code in ("PTG", "PTC", "PBL", "NP", "ON", "OFF"):
            if upper.startswith(code):
                terms[code] = cleaned
                break

    categories: dict[str, float | None] = {}
    for _, row in categories_raw.iloc[1:].iterrows():
        if len(row) < 1 or pd.isna(row.iloc[0]):
            continue

        name = str(row.iloc[0]).strip()
        value = row.iloc[1] if len(row) > 1 else None

        try:
            rate = float(value) if not pd.isna(value) else None
        except (TypeError, ValueError):
            rate = None

        categories[name] = rate

    return {
        "available": True,
        "error": None,
        "phones": pd.DataFrame(phone_rows),
        "tablets": pd.DataFrame(tablet_rows),
        "terms": terms,
        "categories": categories,
        "sheet_names": workbook_sheet_names,
    }


def _manual_default_state() -> dict[str, Any]:
    return {
        "device_type": None,
        "model": None,
        "power_on": None,
        "lcd_good": None,
        "glass_cracked": None,
        "cloud_status": None,
        "weight_lb": None,
        "weight_skipped": False,
        "battery_type": None,
        "notes": [],
    }


def _manual_model_catalog(
    pricing_data: dict[str, Any],
    component_counts: dict[str, Any],
    specifications: dict[str, Any],
    metals_data: dict[str, Any],
) -> list[tuple[str, str]]:
    catalog: dict[str, str] = {}

    for table_name in ("phones", "tablets"):
        dataframe = pricing_data.get(table_name, pd.DataFrame())

        if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
            for model in dataframe["model"].dropna().astype(str).unique():
                catalog[normalize_model_key(model)] = model

    for dataset in (component_counts, specifications, metals_data):
        for key, value in dataset.items():
            display_name = (
                value.get("model")
                if isinstance(value, dict)
                else None
            ) or pretty_class_name(key)
            catalog[normalize_model_key(key)] = str(display_name)

    return sorted(
        catalog.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )


def _manual_find_model(
    text: str,
    catalog: list[tuple[str, str]],
) -> str | None:
    normalized_text = normalize_model_key(text)

    # Prefer explicit containment and longest model first so that
    # "iPhone 15 Pro Max" does not collapse into "iPhone 15 Pro".
    for model_key, display_name in catalog:
        if model_key and model_key in normalized_text:
            return display_name

    # Conservative fuzzy fallback for short answers such as model-only replies.
    best_name = None
    best_score = 0.0

    for model_key, display_name in catalog:
        score = SequenceMatcher(
            None,
            normalized_text,
            model_key,
        ).ratio()

        if score > best_score:
            best_score = score
            best_name = display_name

    return best_name if best_score >= 0.86 else None


def _parse_yes_no(text: str) -> bool | None:
    value = text.strip().lower()

    negative = (
        "no",
        "nope",
        "false",
        "doesn't",
        "does not",
        "dont",
        "don't",
        "not ",
        "won't",
        "wont",
    )
    positive = (
        "yes",
        "yeah",
        "yep",
        "true",
        "works",
        "working",
        "good",
    )

    if any(token in value for token in negative):
        return False

    if any(token in value for token in positive):
        return True

    return None


def _extract_weight_lb(text: str) -> float | None:
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(lb|lbs|pounds?|kg|g|grams?|oz|ounces?)\b",
        text.lower(),
    )

    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)

    if unit in {"lb", "lbs", "pound", "pounds"}:
        return value
    if unit == "kg":
        return value * 2.2046226218
    if unit in {"g", "gram", "grams"}:
        return value / 453.59237
    if unit in {"oz", "ounce", "ounces"}:
        return value / 16.0

    return None


def _manual_apply_message(
    text: str,
    state: dict[str, Any],
    catalog: list[tuple[str, str]],
) -> None:
    """Extract as many useful facts as possible from one free-text message."""
    lower = text.lower().strip()

    model = _manual_find_model(text, catalog)
    if model:
        state["model"] = model

        if "ipad" in model.lower() or "tablet" in model.lower():
            state["device_type"] = "tablet"
        else:
            state["device_type"] = "phone"

    if state.get("device_type") is None:
        if "tablet" in lower or "ipad" in lower:
            state["device_type"] = "tablet"
        elif "phone" in lower or "iphone" in lower or "samsung" in lower:
            state["device_type"] = "phone"

    # Power state
    if any(
        phrase in lower
        for phrase in (
            "no power",
            "dead phone",
            "dead device",
            "won't turn on",
            "wont turn on",
            "doesn't turn on",
            "does not turn on",
            "not turning on",
        )
    ):
        state["power_on"] = False
    elif any(
        phrase in lower
        for phrase in (
            "power on",
            "powers on",
            "turns on",
            "turn on",
            "boots",
            "working phone",
            "working device",
        )
    ):
        state["power_on"] = True

    # LCD/display state
    if any(
        phrase in lower
        for phrase in (
            "bad lcd",
            "broken lcd",
            "bad display",
            "broken display",
            "black screen",
            "display lines",
            "screen lines",
            "lcd bad",
        )
    ):
        state["lcd_good"] = False
    elif any(
        phrase in lower
        for phrase in (
            "good lcd",
            "lcd good",
            "good display",
            "display good",
            "screen works",
            "screen working",
        )
    ):
        state["lcd_good"] = True

    # Glass condition
    if any(
        phrase in lower
        for phrase in (
            "no crack",
            "not cracked",
            "good glass",
            "glass good",
            "no broken glass",
        )
    ):
        state["glass_cracked"] = False
    elif any(
        phrase in lower
        for phrase in (
            "cracked glass",
            "glass cracked",
            "screen cracked",
            "cracked screen",
            "broken glass",
        )
    ):
        state["glass_cracked"] = True

    # iCloud / MDM state
    if any(
        phrase in lower
        for phrase in (
            "icloud off",
            "mdm off",
            "icloud unlocked",
            "activation lock off",
        )
    ):
        state["cloud_status"] = "off"
    elif any(
        phrase in lower
        for phrase in (
            "icloud on",
            "mdm on",
            "icloud locked",
            "activation locked",
            "activation lock on",
        )
    ):
        state["cloud_status"] = "on"

    if "internal battery" in lower or "battery internal" in lower:
        state["battery_type"] = "internal"
    elif "external battery" in lower or "battery external" in lower:
        state["battery_type"] = "external"

    weight = _extract_weight_lb(text)
    if weight is not None:
        state["weight_lb"] = weight
        state["weight_skipped"] = False

    if lower in {"skip", "skip weight", "don't know", "dont know", "unknown"}:
        state["weight_skipped"] = True

    state["notes"].append(text.strip())


def _manual_condition_code(
    state: dict[str, Any],
) -> str | None:
    if state.get("power_on") is False:
        return "NP"

    if state.get("power_on") is not True:
        return None

    if state.get("lcd_good") is False:
        return "PBL"

    if state.get("lcd_good") is not True:
        return None

    if state.get("glass_cracked") is True:
        return "PTC"

    if state.get("glass_cracked") is False:
        return "PTG"

    return None


def _manual_pricing_match(
    state: dict[str, Any],
    pricing_data: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "condition_code": _manual_condition_code(state),
        "matched_model": None,
        "exact_value": None,
        "table": None,
        "cloud_status": state.get("cloud_status"),
        "commodity_name": None,
        "commodity_rate": None,
        "commodity_estimate": None,
    }

    model = state.get("model")
    code = result["condition_code"]

    if model and code and pricing_data.get("available"):
        table_name = (
            "tablets"
            if state.get("device_type") == "tablet"
            else "phones"
        )
        dataframe = pricing_data.get(table_name, pd.DataFrame())
        model_key = normalize_model_key(model)

        if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
            matches = dataframe[
                dataframe["model_key"] == model_key
            ]

            if table_name == "tablets" and not matches.empty:
                cloud_status = state.get("cloud_status")
                if cloud_status in {"on", "off"}:
                    matches = matches[
                        matches["cloud_status"] == cloud_status
                    ]

            if not matches.empty:
                row = matches.iloc[0]
                result["matched_model"] = row["model"]
                result["table"] = table_name
                value = row.get(code)

                if pd.notna(value):
                    try:
                        result["exact_value"] = float(value)
                    except (TypeError, ValueError):
                        pass

    # Optional commodity fallback from the workbook's Categories wprice sheet.
    categories = pricing_data.get("categories", {}) or {}
    category_name = None

    if state.get("device_type") == "tablet":
        category_name = (
            "iPad"
            if "ipad" in str(state.get("model", "")).lower()
            else "Tablet"
        )
    elif state.get("device_type") == "phone":
        if state.get("battery_type") == "external":
            category_name = "Phone external batt"
        else:
            category_name = "Phone internal batt"

    if category_name:
        rate = categories.get(category_name)
        result["commodity_name"] = category_name
        result["commodity_rate"] = rate

        if rate is not None and state.get("weight_lb") is not None:
            result["commodity_estimate"] = (
                float(rate) * float(state["weight_lb"])
            )

    return result


def _manual_needs_commodity_fallback(
    state: dict[str, Any],
    pricing_data: dict[str, Any],
) -> bool:
    pricing = _manual_pricing_match(state, pricing_data)
    return (
        pricing.get("condition_code") is not None
        and pricing.get("exact_value") is None
    )


def _manual_next_question(
    state: dict[str, Any],
    pricing_data: dict[str, Any],
) -> str | None:
    if not state.get("model"):
        return (
            "What is the smartphone/tablet brand and model? "
            "For example: iPhone 14, iPhone 15 Pro, Galaxy S24 Ultra, or iPad 9."
        )

    if not state.get("device_type"):
        return "Is this a phone or a tablet?"

    if state.get("power_on") is None:
        return "Does the device power on?"

    if state.get("power_on") is True and state.get("lcd_good") is None:
        return (
            "Is the LCD/display working normally? "
            "Tell me if it has a bad LCD, black screen, lines, or other display failure."
        )

    if (
        state.get("power_on") is True
        and state.get("lcd_good") is True
        and state.get("glass_cracked") is None
    ):
        return "Is the front glass cracked or broken?"

    if (
        state.get("device_type") == "tablet"
        and state.get("cloud_status") is None
    ):
        return "Is iCloud/MDM ON or OFF?"

    if _manual_needs_commodity_fallback(state, pricing_data):
        if state.get("device_type") == "phone" and not state.get("battery_type"):
            return (
                "The workbook has no exact price for this model/condition. "
                "For the commodity-rate fallback, is the phone battery internal or external?"
            )

        if state.get("weight_lb") is None and not state.get("weight_skipped"):
            return (
                "The workbook has no exact model price for this condition. "
                "If you want a commodity-rate fallback, provide the device weight "
                "(for example 0.45 lb or 205 g), or reply 'skip'."
            )

    return None


def _manual_answer_current_question(
    text: str,
    state: dict[str, Any],
    question: str | None,
) -> None:
    """Interpret short answers according to the question currently being asked."""
    if not question:
        return

    lower_question = question.lower()
    lower_text = text.lower().strip()

    if "power on" in lower_question and state.get("power_on") is None:
        state["power_on"] = _parse_yes_no(text)

    elif "lcd/display" in lower_question and state.get("lcd_good") is None:
        state["lcd_good"] = _parse_yes_no(text)

    elif "front glass" in lower_question and state.get("glass_cracked") is None:
        answer = _parse_yes_no(text)
        # Here "yes" means yes, it is cracked.
        if answer is not None:
            state["glass_cracked"] = answer

    elif "icloud/mdm" in lower_question and state.get("cloud_status") is None:
        if "off" in lower_text or "unlocked" in lower_text:
            state["cloud_status"] = "off"
        elif "on" in lower_text or "locked" in lower_text:
            state["cloud_status"] = "on"

    elif "battery internal or external" in lower_question:
        if "external" in lower_text:
            state["battery_type"] = "external"
        elif "internal" in lower_text:
            state["battery_type"] = "internal"


def _manual_summary_text(
    state: dict[str, Any],
    pricing_data: dict[str, Any],
    component_counts: dict[str, Any],
    metals_data: dict[str, Any],
) -> str:
    pricing = _manual_pricing_match(state, pricing_data)
    code = pricing.get("condition_code") or "Unknown"
    terms = pricing_data.get("terms", {}) or {}
    condition_text = terms.get(code, code)

    if pricing.get("exact_value") is not None:
        price_text = f"{pricing['exact_value']:.2f} from the workbook"
    elif pricing.get("commodity_estimate") is not None:
        price_text = (
            f"{pricing['commodity_estimate']:.2f} using the workbook commodity rate "
            f"({pricing['commodity_rate']} × {state['weight_lb']:.3f} lb)"
        )
    else:
        price_text = "No exact workbook price is available for the supplied model/condition"

    part_data = lookup_component_counts(
        str(state.get("model", "")),
        component_counts,
    )
    part_count = len(part_data.get("components", [])) if part_data else 0

    metal_data = lookup_valuable_metals(
        str(state.get("model", "")),
        metals_data,
    )
    metal_count = len(metal_data.get("valuable_metals", {})) if metal_data else 0

    return (
        f"I have enough information to prepare the estimate. "
        f"Model: **{state.get('model')}**. Condition: **{condition_text}**. "
        f"Pricing result: **{price_text}**. "
        f"I also found **{part_count} component records** and "
        f"**{metal_count} valuable-metal records** for this model. "
        "The detailed report is shown below."
    )


def render_manual_valuation_report(
    state: dict[str, Any],
    pricing_data: dict[str, Any],
    component_dataframe: pd.DataFrame,
    component_counts: dict[str, Any],
    specifications: dict[str, Any],
    metals_data: dict[str, Any],
) -> None:
    model = state.get("model")
    if not model:
        return

    pricing = _manual_pricing_match(state, pricing_data)
    code = pricing.get("condition_code")

    st.markdown("---")
    st.subheader("Manual valuation report")

    summary_cols = st.columns(4)
    summary_cols[0].metric("Model", model)
    summary_cols[1].metric("Condition code", code or "—")

    if pricing.get("exact_value") is not None:
        summary_cols[2].metric(
            "Workbook price",
            f"{pricing['exact_value']:.2f}",
        )
        summary_cols[3].metric(
            "Price source",
            "Model / condition",
        )
    elif pricing.get("commodity_estimate") is not None:
        summary_cols[2].metric(
            "Commodity estimate",
            f"{pricing['commodity_estimate']:.2f}",
        )
        summary_cols[3].metric(
            "Commodity rate",
            str(pricing.get("commodity_rate", "—")),
        )
    else:
        summary_cols[2].metric("Workbook price", "Unavailable")
        summary_cols[3].metric(
            "Commodity rate",
            str(pricing.get("commodity_rate") or "—"),
        )

    if code:
        description = (pricing_data.get("terms", {}) or {}).get(code)
        if description:
            st.caption(f"Workbook condition definition: {description}")

    if pricing.get("exact_value") is None:
        st.info(
            "The uploaded pricing workbook does not contain an exact value for "
            "this model/condition. Any commodity estimate shown above is calculated "
            "from the workbook's Categories wprice rate multiplied by the weight "
            "you supplied. The workbook does not explicitly label the rate/weight units, "
            "so verify those units before using the value operationally."
        )

    render_phone_specification(model, specifications)
    render_component_counts(model, component_counts)

    # Fallback to source-derived component names if the count/part-number JSON
    # has no entry for this device.
    part_data = lookup_component_counts(model, component_counts)
    if not part_data:
        matches = lookup_components(model, component_dataframe)
        if not matches.empty:
            st.subheader("Components found in source-guide dataset")
            component_col = next(
                (
                    column
                    for column in ("component_name", "component", "pcb_type")
                    if column in matches.columns
                ),
                None,
            )
            if component_col:
                fallback = (
                    matches[[component_col]]
                    .dropna()
                    .drop_duplicates()
                    .rename(columns={component_col: "Component"})
                )
                st.dataframe(
                    fallback,
                    use_container_width=True,
                    hide_index=True,
                )

    render_valuable_metals(model, metals_data)

    with st.expander("Information supplied by user"):
        st.json(
            {
                "device_type": state.get("device_type"),
                "model": state.get("model"),
                "power_on": state.get("power_on"),
                "lcd_good": state.get("lcd_good"),
                "glass_cracked": state.get("glass_cracked"),
                "icloud_mdm": state.get("cloud_status"),
                "weight_lb": state.get("weight_lb"),
                "battery_type": state.get("battery_type"),
            }
        )



def _manual_progress(
    state: dict[str, Any],
    pricing_data: dict[str, Any],
) -> tuple[int, int, float]:
    """Return completed required steps, total required steps, and progress."""

    required = [
        state.get("model") is not None,
        state.get("device_type") is not None,
        state.get("power_on") is not None,
    ]

    if state.get("power_on") is True:
        required.append(state.get("lcd_good") is not None)

        if state.get("lcd_good") is True:
            required.append(
                state.get("glass_cracked") is not None
            )

    if state.get("device_type") == "tablet":
        required.append(
            state.get("cloud_status") is not None
        )

    if _manual_needs_commodity_fallback(
        state,
        pricing_data,
    ):
        if state.get("device_type") == "phone":
            required.append(
                state.get("battery_type") is not None
            )

        required.append(
            state.get("weight_lb") is not None
            or state.get("weight_skipped", False)
        )

    total = max(1, len(required))
    completed = sum(bool(item) for item in required)

    return (
        completed,
        total,
        completed / total,
    )


def _manual_quick_replies(
    question: str | None,
) -> list[tuple[str, str]]:
    """Return friendly quick-reply buttons for the current question."""

    if not question:
        return []

    question_lower = question.lower()

    if "phone or a tablet" in question_lower:
        return [
            ("📱 Phone", "phone"),
            ("💻 Tablet", "tablet"),
        ]

    if "power on" in question_lower:
        return [
            ("✅ Yes, powers on", "yes"),
            ("❌ No power", "no"),
        ]

    if "lcd/display" in question_lower:
        return [
            ("✅ Display is good", "yes"),
            ("❌ Display is bad", "no"),
        ]

    if "front glass" in question_lower:
        return [
            ("💥 Yes, cracked", "yes"),
            ("✨ No cracks", "no"),
        ]

    if "icloud/mdm" in question_lower:
        return [
            ("🔓 OFF / Unlocked", "off"),
            ("🔒 ON / Locked", "on"),
        ]

    if "battery internal or external" in question_lower:
        return [
            ("🔋 Internal", "internal battery"),
            ("🔌 External", "external battery"),
        ]

    if "device weight" in question_lower:
        return [
            ("Skip weight", "skip"),
        ]

    return []


def _manual_process_user_message(
    user_text: str,
    state: dict[str, Any],
    messages: list[dict[str, str]],
    catalog: list[str],
    pricing_data: dict[str, Any],
) -> None:
    """Process one conversational answer and append the assistant response."""

    if not user_text or not user_text.strip():
        return

    cleaned_text = user_text.strip()

    messages.append(
        {
            "role": "user",
            "content": cleaned_text,
        }
    )

    previous_question = (
        st.session_state.manual_last_question
    )

    _manual_apply_message(
        cleaned_text,
        state,
        catalog,
    )

    _manual_answer_current_question(
        cleaned_text,
        state,
        previous_question,
    )

    next_question = _manual_next_question(
        state,
        pricing_data,
    )

    if next_question:
        assistant_text = next_question
    else:
        assistant_text = _manual_summary_text(
            state,
            pricing_data,
            st.session_state.get(
                "manual_component_counts",
                {},
            ),
            st.session_state.get(
                "manual_metals_data",
                {},
            ),
        )

    messages.append(
        {
            "role": "assistant",
            "content": assistant_text,
        }
    )

    st.session_state.manual_last_question = (
        next_question
    )
    st.session_state.manual_valuation_state = state
    st.session_state.manual_valuation_messages = (
        messages
    )


def render_manual_valuation_assistant(
    pricing_data: dict[str, Any],
    component_dataframe: pd.DataFrame,
    component_counts: dict[str, Any],
    specifications: dict[str, Any],
    metals_data: dict[str, Any],
) -> None:
    # Make these available to the message-processing helper.
    st.session_state.manual_component_counts = (
        component_counts
    )
    st.session_state.manual_metals_data = metals_data

    # --------------------------------------------------------
    # Initialize conversation state
    # --------------------------------------------------------

    if "manual_valuation_state" not in st.session_state:
        st.session_state.manual_valuation_state = (
            _manual_default_state()
        )

    if (
        "manual_valuation_messages"
        not in st.session_state
    ):
        st.session_state.manual_valuation_messages = []

    if "manual_last_question" not in st.session_state:
        st.session_state.manual_last_question = None

    state = st.session_state.manual_valuation_state
    messages = (
        st.session_state.manual_valuation_messages
    )

    catalog = _manual_model_catalog(
        pricing_data,
        component_counts,
        specifications,
        metals_data,
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="prediction-card">
            <div class="prediction-label">
                Conversational valuation
            </div>
            <div class="prediction-name">
                💬 Device Valuation Assistant
            </div>
            <div class="small-note">
                Chat naturally with the assistant. It will ask one
                question at a time and use your answers to estimate
                device value, identify parts, and summarize valuable
                metals.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not pricing_data.get("available"):
        st.error(
            pricing_data.get("error")
            or "Pricing workbook could not be loaded."
        )

        available_sheets = pricing_data.get(
            "sheet_names",
            [],
        )

        if available_sheets:
            st.caption(
                "Worksheets detected: "
                + ", ".join(available_sheets)
            )

    # --------------------------------------------------------
    # Start greeting
    # --------------------------------------------------------

    if not messages:
        greeting = (
            "Hi! 👋 I can estimate the value of your device "
            "without using an image. I’ll ask a few short "
            "questions, then I’ll show the estimated price, "
            "specifications, component names and quantities, "
            "part numbers, and estimated valuable-metal content.\n\n"
            "**What is the phone or tablet model?**"
        )

        messages.append(
            {
                "role": "assistant",
                "content": greeting,
            }
        )

        st.session_state.manual_last_question = (
            "What is the smartphone/tablet brand and model?"
        )

    # --------------------------------------------------------
    # Conversation toolbar
    # --------------------------------------------------------

    toolbar_left, toolbar_middle, toolbar_right = (
        st.columns([1, 3, 1])
    )

    with toolbar_left:
        if st.button(
            "↻ New chat",
            key="manual_reset",
            use_container_width=True,
        ):
            st.session_state.manual_valuation_state = (
                _manual_default_state()
            )
            st.session_state.manual_valuation_messages = []
            st.session_state.manual_last_question = None
            st.rerun()

    completed, total, progress = _manual_progress(
        state,
        pricing_data,
    )

    with toolbar_middle:
        st.progress(
            progress,
            text=(
                f"Device information: "
                f"{completed}/{total} steps complete"
            ),
        )

    with toolbar_right:
        current_condition = (
            _manual_condition_code(state)
            or "Pending"
        )

        st.metric(
            "Condition",
            current_condition,
        )

    # --------------------------------------------------------
    # Compact collected-information panel
    # --------------------------------------------------------

    collected = []

    if state.get("model"):
        collected.append(
            f"**Model:** {state['model']}"
        )

    if state.get("device_type"):
        collected.append(
            f"**Type:** {state['device_type'].title()}"
        )

    if state.get("power_on") is not None:
        collected.append(
            "**Power:** "
            + (
                "On"
                if state["power_on"]
                else "No power"
            )
        )

    if state.get("lcd_good") is not None:
        collected.append(
            "**Display:** "
            + (
                "Good"
                if state["lcd_good"]
                else "Bad"
            )
        )

    if state.get("glass_cracked") is not None:
        collected.append(
            "**Glass:** "
            + (
                "Cracked"
                if state["glass_cracked"]
                else "Good"
            )
        )

    if state.get("cloud_status"):
        collected.append(
            f"**iCloud/MDM:** "
            f"{state['cloud_status'].upper()}"
        )

    if collected:
        with st.expander(
            "Device information collected so far",
            expanded=False,
        ):
            st.markdown("  \n".join(collected))

    # --------------------------------------------------------
    # Chat conversation
    # --------------------------------------------------------

    st.markdown("### Conversation")

    chat_container = st.container(
        height=520,
        border=True,
    )

    with chat_container:
        for message in messages:
            role = message.get(
                "role",
                "assistant",
            )

            avatar = (
                "🤖"
                if role == "assistant"
                else "👤"
            )

            with st.chat_message(
                role,
                avatar=avatar,
            ):
                st.markdown(
                    message.get(
                        "content",
                        "",
                    )
                )

    current_question = (
        st.session_state.manual_last_question
    )

    # --------------------------------------------------------
    # Quick reply buttons
    # --------------------------------------------------------

    quick_replies = _manual_quick_replies(
        current_question
    )

    if quick_replies:
        st.caption("Quick reply")

        reply_columns = st.columns(
            len(quick_replies)
        )

        for index, (
            label,
            response_text,
        ) in enumerate(quick_replies):
            with reply_columns[index]:
                if st.button(
                    label,
                    key=(
                        "manual_quick_"
                        f"{index}_"
                        f"{len(messages)}"
                    ),
                    use_container_width=True,
                ):
                    _manual_process_user_message(
                        response_text,
                        state,
                        messages,
                        catalog,
                        pricing_data,
                    )
                    st.rerun()

    # --------------------------------------------------------
    # ChatGPT-style free-text input
    # --------------------------------------------------------

    if current_question:
        placeholder = current_question
    else:
        placeholder = (
            "Ask another question or add more device details..."
        )

    user_text = st.chat_input(
        placeholder,
        key="manual_valuation_chat_input",
    )

    if user_text:
        _manual_process_user_message(
            user_text,
            state,
            messages,
            catalog,
            pricing_data,
        )
        st.rerun()

    # --------------------------------------------------------
    # Final valuation report
    # --------------------------------------------------------

    if (
        _manual_next_question(
            state,
            pricing_data,
        )
        is None
    ):
        st.divider()

        st.markdown(
            "## Valuation result"
        )

        render_manual_valuation_report(
            state,
            pricing_data,
            component_dataframe,
            component_counts,
            specifications,
            metals_data,
        )

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

    model_path_override = st.text_input(
        "Checkpoint path (optional)",
        value=str(DEFAULT_MODEL_PATH),
        help=(
            "Leave this blank to download best_model.pth automatically "
            "from Hugging Face. Enter a local checkpoint path only when "
            "running the app locally."
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

    component_counts_path = st.text_input(
        "Component quantity JSON path",
        value=str(DEFAULT_COMPONENT_COUNTS_PATH),
        help=(
            "JSON containing component names, quantities, dataset part "
            "numbers, and verified manufacturer part numbers."
        ),
    )

    metals_path = st.text_input(
        "Valuable metals JSON path",
        value=str(DEFAULT_METALS_PATH),
        help=(
            "JSON containing estimated valuable-metal quantities for each "
            "smartphone model."
        ),
    )

    pricing_path = st.text_input(
        "Pricing Excel path",
        value=str(DEFAULT_PRICING_PATH),
        help=(
            "Excel workbook containing model prices for PTG, PTC, PBL, "
            "NP, cloud-status, and commodity pricing criteria."
        ),
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
    effective_model_path = (
        model_path_override.strip()
        if model_path_override.strip()
        else get_model_path()
    )

    model_bundle = load_classifier(
        effective_model_path
    )
    model_ready = True
    model_error = None

except Exception as error:
    effective_model_path = None
    model_bundle = None
    model_ready = False
    model_error = error

component_dataframe = load_component_dataset(components_path)
phone_specifications = load_phone_specifications(specifications_path)
component_counts = load_component_counts(component_counts_path)
valuable_metals_data = load_valuable_metals(metals_path)
pricing_data = load_pricing_workbook(pricing_path)

status_columns = st.columns(8)

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

with status_columns[5]:
    st.metric(
        "Component-count models",
        len(component_counts),
    )

with status_columns[6]:
    st.metric(
        "Metal records",
        len(valuable_metals_data),
    )

with status_columns[7]:
    st.metric(
        "Pricing workbook",
        "Ready"
        if pricing_data.get("available")
        else "Unavailable",
    )

if not model_ready:
    st.warning(
        "AI detector is currently unavailable. "
        "The Manual Valuation Assistant remains fully available "
        "without the image-classification model."
    )

    with st.expander("AI model loading details"):
        st.code(str(model_error))

mode = st.tabs(
    [
        "Upload image",
        "Camera snapshot",
        "Live camera",
        "Manual valuation assistant",
        "Model information",
    ]
)

with mode[0]:
    if not model_ready:
        st.error(
            "AI detector is unavailable. "
            "Use the Manual valuation assistant tab instead."
        )
    else:
        uploaded_files = st.file_uploader(
            "Upload one or more smartphone images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help=(
                "You can select several images and analyze "
                "them in one batch."
            ),
        )

        if uploaded_files:
            st.caption(
                f"{len(uploaded_files)} image(s) selected"
            )

            for image_index, uploaded_file in enumerate(
                uploaded_files,
                start=1,
            ):
                try:
                    uploaded_image = Image.open(
                        uploaded_file
                    ).convert("RGB")
                except Exception as exc:
                    st.error(
                        f"Could not open "
                        f"{uploaded_file.name}: {exc}"
                    )
                    continue

                with st.container(border=True):
                    st.markdown(
                        f"### Image {image_index}: "
                        f"{uploaded_file.name}"
                    )

                    first, second = st.columns(
                        [1.05, 0.95],
                        gap="large",
                    )

                    with first:
                        st.image(
                            uploaded_image,
                            caption=uploaded_file.name,
                            use_container_width=True,
                        )

                    with second:
                        with st.spinner(
                            f"Analyzing "
                            f"{uploaded_file.name}..."
                        ):
                            predictions = predict_pil_image(
                                uploaded_image,
                                model_bundle,
                                top_k=top_k,
                            )

                        render_prediction(
                            predictions,
                            component_dataframe,
                            component_counts,
                            phone_specifications,
                            valuable_metals_data,
                            confidence_threshold,
                            image=uploaded_image,
                        )
        else:
            st.info(
                "Upload one or more images to begin."
            )

with mode[1]:
    if not model_ready:
        st.error(
            "AI detector is unavailable. "
            "Use the Manual valuation assistant tab instead."
        )
    else:
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
                camera_image = Image.open(
                    camera_file
                ).convert("RGB")

                with st.spinner(
                    "Analyzing camera image..."
                ):
                    predictions = predict_pil_image(
                        camera_image,
                        model_bundle,
                        top_k=top_k,
                    )

                render_prediction(
                    predictions,
                    component_dataframe,
                    component_counts,
                    phone_specifications,
                    valuable_metals_data,
                    confidence_threshold,
                    image=camera_image,
                )
            else:
                st.info(
                    "Allow camera access and "
                    "take a picture."
                )

with mode[2]:
    if not model_ready:
        st.error("AI detector is unavailable. Use the Manual valuation assistant tab instead.")
    else:
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
    render_manual_valuation_assistant(
        pricing_data,
        component_dataframe,
        component_counts,
        phone_specifications,
        valuable_metals_data,
    )

with mode[4]:
    if not model_ready:
        st.info("The AI detector model is not loaded. Manual valuation remains available.")
    else:
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
