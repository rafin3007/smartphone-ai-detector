from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

def pil_to_cv2(image):
    image_np = np.array(image.convert("RGB"))
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


def _resize_for_analysis(image_cv, max_side=1400):
    h, w = image_cv.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image_cv = cv2.resize(
            image_cv,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image_cv, scale


def _analysis_roi(gray, border_ratio=0.07):
    h, w = gray.shape
    bx = max(2, int(w * border_ratio))
    by = max(2, int(h * border_ratio))
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[by:h - by, bx:w - bx] = 255
    return mask, (bx, by, w - bx, h - by)


def _morphological_skeleton(binary):
    binary = (binary > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    working = binary.copy()
    while cv2.countNonZero(working) > 0:
        opened = cv2.morphologyEx(working, cv2.MORPH_OPEN, element)
        residue = cv2.subtract(working, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        working = cv2.erode(working, element)
    return skeleton


def _neighbor_statistics(skeleton):
    binary = (skeleton > 0).astype(np.uint8)
    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(binary, -1, neighbor_kernel) - binary
    endpoints = int(np.sum((binary == 1) & (neighbors == 1)))
    junctions = int(np.sum((binary == 1) & (neighbors >= 3)))
    return endpoints, junctions


def assess_image_quality(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    dark_ratio = float(np.mean(gray < 20))
    glare_ratio = float(np.mean(gray > 248))

    warnings = []
    if blur_score < 45:
        warnings.append("Image is blurry")
    if brightness < 35:
        warnings.append("Image is too dark")
    if brightness > 225:
        warnings.append("Image is overexposed")
    if glare_ratio > 0.12:
        warnings.append("Strong glare may look like scratches")
    if dark_ratio > 0.70:
        warnings.append("Most of the phone is nearly black")

    quality_score = 1.0
    quality_score -= min(0.35, max(0.0, (60 - blur_score) / 100))
    quality_score -= min(0.25, glare_ratio * 1.5)
    if brightness < 35 or brightness > 225:
        quality_score -= 0.20
    quality_score = float(np.clip(quality_score, 0.0, 1.0))

    return {
        "quality_score": round(quality_score, 3),
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "dark_ratio": round(dark_ratio, 4),
        "glare_ratio": round(glare_ratio, 4),
        "warnings": warnings,
    }


def _oriented_line_response(gray):
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    responses = []

    for length in (9, 15, 23):
        base = np.zeros((length, length), dtype=np.uint8)
        cv2.line(base, (1, length // 2), (length - 2, length // 2), 1, 1)

        for angle in (0, 30, 60, 90, 120, 150):
            matrix = cv2.getRotationMatrix2D(
                ((length - 1) / 2, (length - 1) / 2), angle, 1.0
            )
            kernel = cv2.warpAffine(base, matrix, (length, length))
            kernel = (kernel > 0).astype(np.uint8)
            if kernel.sum() < 2:
                continue
            blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
            tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
            responses.append(cv2.max(blackhat, tophat))

    response = np.maximum.reduce(responses) if responses else enhanced
    response = cv2.GaussianBlur(response, (3, 3), 0)
    return enhanced, response


def _build_linear_damage_mask(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    roi_mask, bounds = _analysis_roi(gray)
    enhanced, line_response = _oriented_line_response(gray)

    roi_values = line_response[roi_mask > 0]
    percentile = 93 if roi_values.size else 255
    threshold_value = max(16, int(np.percentile(roi_values, percentile)))
    _, response_mask = cv2.threshold(
        line_response, threshold_value, 255, cv2.THRESH_BINARY
    )

    edges = cv2.Canny(enhanced, 45, 135, L2gradient=True)
    candidate = cv2.bitwise_and(
        cv2.bitwise_or(response_mask, edges), roi_mask
    )

    # Suppress large highlights and textured blobs.
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    glare = cv2.inRange(hsv, (0, 0, 235), (180, 55, 255))
    glare = cv2.dilate(glare, np.ones((7, 7), np.uint8), iterations=1)
    candidate[glare > 0] = 0

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return candidate, enhanced, line_response, bounds


def _classify_linear_components(candidate_mask):
    h, w = candidate_mask.shape
    image_area = h * w
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask, connectivity=8
    )

    crack_mask = np.zeros_like(candidate_mask)
    scratch_mask = np.zeros_like(candidate_mask)
    crack_features = []
    scratch_features = []

    for label in range(1, num_labels):
        x, y, cw, ch, area = stats[label]
        if area < max(8, image_area * 0.000015):
            continue
        if area > image_area * 0.08:
            continue

        component = np.zeros_like(candidate_mask)
        component[labels == label] = 255
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, False))
        span = float(np.hypot(cw, ch))
        aspect = max(cw, ch) / max(1, min(cw, ch))
        fill_ratio = area / max(1, cw * ch)
        skeleton = _morphological_skeleton(component)
        skeleton_length = int(np.count_nonzero(skeleton))
        endpoints, junctions = _neighbor_statistics(skeleton)

        # Ignore very long, perfectly straight phone/frame edges.
        rect = cv2.minAreaRect(contour)
        rect_w, rect_h = rect[1]
        straightness = max(rect_w, rect_h) / max(1.0, perimeter)
        if span > 0.80 * max(h, w) and straightness > 0.65:
            continue

        feature = {
            "area": int(area),
            "span": round(span, 2),
            "aspect": round(aspect, 2),
            "fill_ratio": round(fill_ratio, 3),
            "skeleton_length": skeleton_length,
            "endpoints": endpoints,
            "junctions": junctions,
        }

        is_branching = junctions >= 2 or endpoints >= 4
        is_long = skeleton_length >= max(18, int(min(h, w) * 0.035))
        is_thin = fill_ratio < 0.52

        if is_long and is_thin and is_branching:
            crack_mask = cv2.bitwise_or(crack_mask, component)
            crack_features.append(feature)
        elif is_long and aspect >= 2.8 and junctions <= 2:
            scratch_mask = cv2.bitwise_or(scratch_mask, component)
            scratch_features.append(feature)

    return crack_mask, scratch_mask, crack_features, scratch_features


def detect_cracks(image_cv):
    candidate, _, _, _ = _build_linear_damage_mask(image_cv)
    crack_mask, _, features, _ = _classify_linear_components(candidate)
    skeleton = _morphological_skeleton(crack_mask)
    line_length = int(np.count_nonzero(skeleton))
    _, junctions = _neighbor_statistics(skeleton)
    component_count = len(features)

    h, w = crack_mask.shape
    normalized_length = line_length / max(1, np.hypot(h, w))
    score = (
        component_count * 0.16
        + junctions * 0.055
        + normalized_length * 0.18
    )
    confidence = float(np.clip(score, 0.0, 1.0))
    cracked = confidence >= 0.52

    return cracked, component_count, round(confidence, 3), crack_mask


def detect_scratches(image_cv):
    candidate, _, _, _ = _build_linear_damage_mask(image_cv)
    _, scratch_mask, _, features = _classify_linear_components(candidate)
    skeleton = _morphological_skeleton(scratch_mask)
    total_length = int(np.count_nonzero(skeleton))
    scratch_count = len(features)

    h, w = scratch_mask.shape
    coverage = float(np.mean(scratch_mask > 0))
    normalized_length = total_length / max(1, np.hypot(h, w))
    score = (
        scratch_count * 0.10
        + normalized_length * 0.13
        + coverage * 5.0
    )
    confidence = float(np.clip(score, 0.0, 1.0))

    if confidence < 0.22:
        severity = "None"
    elif confidence < 0.42:
        severity = "Minor"
    elif confidence < 0.68:
        severity = "Moderate"
    else:
        severity = "Severe"

    return (
        severity != "None",
        round(confidence, 3),
        severity,
        scratch_count,
        scratch_mask,
    )


def detect_broken_display(image_cv, crack_confidence=0.0):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    x1, x2 = int(w * 0.16), int(w * 0.84)
    y1, y2 = int(h * 0.16), int(h * 0.84)
    region = gray[y1:y2, x1:x2]

    dark_ratio = float(np.mean(region < 22))
    bright_ratio = float(np.mean(region > 248))
    laplacian = cv2.Laplacian(region, cv2.CV_32F)
    abnormal_line_ratio = float(np.mean(np.abs(laplacian) > 55))

    # A switched-off black screen alone is not considered broken.
    broken_score = 0.0
    if crack_confidence >= 0.52:
        broken_score += crack_confidence * 0.55
    if abnormal_line_ratio > 0.09:
        broken_score += min(0.35, abnormal_line_ratio * 2.0)
    if bright_ratio > 0.45 and abnormal_line_ratio > 0.08:
        broken_score += 0.20
    if dark_ratio > 0.72 and crack_confidence > 0.60:
        broken_score += 0.15

    confidence = float(np.clip(broken_score, 0.0, 1.0))
    broken = confidence >= 0.58

    if broken:
        message = "Display damage pattern detected"
    elif dark_ratio > 0.72:
        message = "Screen may simply be switched off; not marked broken"
    else:
        message = "No strong broken-display evidence"

    return broken, confidence, message


def detect_body_damage(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 55, 155)
    roi_mask, _ = _analysis_roi(gray, border_ratio=0.04)
    edges = cv2.bitwise_and(edges, roi_mask)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    irregular_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < h * w * 0.0008 or area > h * w * 0.12:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        x, y, cw, ch = cv2.boundingRect(contour)
        aspect = max(cw, ch) / max(1, min(cw, ch))
        if circularity < 0.22 and aspect < 7:
            irregular_count += 1

    confidence = float(np.clip(irregular_count / 10.0, 0.0, 1.0))
    return confidence >= 0.65, round(confidence, 3)


def detect_camera_lens_damage(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    h, w = gray.shape
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, int(min(h, w) * 0.06)),
        param1=100,
        param2=34,
        minRadius=max(7, int(min(h, w) * 0.012)),
        maxRadius=max(20, int(min(h, w) * 0.12)),
    )

    lens_count = 0
    damage_scores = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        lens_count = len(circles)
        for x, y, radius in circles:
            x1, y1 = max(int(x - radius), 0), max(int(y - radius), 0)
            x2, y2 = min(int(x + radius), w), min(int(y + radius), h)
            region = gray[y1:y2, x1:x2]
            if region.size == 0:
                continue
            edges = cv2.Canny(region, 75, 180)
            edge_density = float(np.mean(edges > 0))
            center_variance = float(np.var(region)) / (255.0 ** 2)
            damage_scores.append(min(1.0, edge_density * 3.0 + center_variance))

    confidence = max(damage_scores, default=0.0)
    return confidence >= 0.62, lens_count, round(confidence, 3)


def calculate_image_condition(
    cracked,
    broken_display,
    scratch_severity,
    body_damage,
    camera_damage,
):
    damage_points = 0
    if cracked:
        damage_points += 3
    if broken_display:
        damage_points += 4
    damage_points += {
        "None": 0,
        "Minor": 1,
        "Moderate": 2,
        "Severe": 3,
    }.get(scratch_severity, 1)
    if body_damage:
        damage_points += 2
    if camera_damage:
        damage_points += 3

    if damage_points == 0:
        condition = "Excellent"
    elif damage_points <= 2:
        condition = "Good"
    elif damage_points <= 5:
        condition = "Fair"
    else:
        condition = "Damaged"
    return condition, damage_points


def analyze_phone_damage(image):
    image_cv = pil_to_cv2(image)
    image_cv, _ = _resize_for_analysis(image_cv)
    quality = assess_image_quality(image_cv)

    cracked, crack_components, crack_confidence, crack_mask = detect_cracks(image_cv)
    scratches, scratch_confidence, scratch_severity, scratch_count, scratch_mask = detect_scratches(image_cv)
    broken_display, display_confidence, display_message = detect_broken_display(
        image_cv, crack_confidence=crack_confidence
    )
    body_damage, body_confidence = detect_body_damage(image_cv)
    camera_damage, lens_count, camera_confidence = detect_camera_lens_damage(image_cv)

    # Reduce false positives when image quality is poor.
    if quality["quality_score"] < 0.45:
        if crack_confidence < 0.72:
            cracked = False
        if scratch_confidence < 0.70:
            scratches = False
            scratch_severity = "None"
        if display_confidence < 0.75:
            broken_display = False

    image_condition, damage_points = calculate_image_condition(
        cracked,
        broken_display,
        scratch_severity,
        body_damage,
        camera_damage,
    )

    damage_result = {
        "cracked_screen": cracked,
        "broken_display": broken_display,
        "visible_scratches": scratches,
        "scratch_severity": scratch_severity,
        "damaged_back_or_body": body_damage,
        "camera_lens_damage": camera_damage,
        "image_condition": image_condition,
        "damage_points": damage_points,
        "damage_confidence": {
            "crack": crack_confidence,
            "scratch": scratch_confidence,
            "broken_display": round(display_confidence, 3),
            "body": body_confidence,
            "camera_lens": camera_confidence,
        },
        "image_quality": quality,
        "debug_scores": {
            "crack_component_count": crack_components,
            "crack_confidence": crack_confidence,
            "display_confidence": round(display_confidence, 3),
            "display_message": display_message,
            "scratch_confidence": scratch_confidence,
            "scratch_severity": scratch_severity,
            "scratch_component_count": scratch_count,
            "body_damage_confidence": body_confidence,
            "detected_camera_lens_count": lens_count,
            "camera_damage_confidence": camera_confidence,
            "image_quality": quality,
        },
    }
    return damage_result, scratch_mask, crack_mask

def get_scratch_overlay(image, scratch_mask):
    image_np = np.array(image.convert("RGB"))
    highlight = image_np.copy()
    highlight[scratch_mask > 0] = [255, 220, 0]
    blended = cv2.addWeighted(image_np, 0.65, highlight, 0.35, 0)
    contours, _ = cv2.findContours(scratch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 140, 0), 1)
    return Image.fromarray(blended)


def get_crack_overlay(image, crack_mask):
    image_np = np.array(image.convert("RGB"))
    highlight = image_np.copy()
    highlight[crack_mask > 0] = [255, 50, 50]
    blended = cv2.addWeighted(image_np, 0.60, highlight, 0.40, 0)
    contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (220, 0, 0), 2)
    return Image.fromarray(blended)
