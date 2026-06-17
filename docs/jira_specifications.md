# Jira Specifications — Document Verification Model

**Jira Ticket:** [FRAUD-1045](https://jira.sunfinance.group/browse/FRAUD-1045)
**Summary:** Document verification model
**Status:** Backlog
**Priority:** Medium
**Assignee:** Ansis Vaišļa

---

## Overview

Build a multi-stage forensic model that processes ID document photos to determine whether a submitted document is a **genuine, physically present, accepted ID type** — rather than a spoofed image, a photo of a screen, or a printed forgery.

Initial target country: **Kenya**

---

## Pipeline Stages

### Stage 1 — ID Cropping

Live photos introduce severe perspective skew, background clutter, random rotations, and occasionally truncated framing.

**Corner Detection & Partial ID Handling**

Train an object detection model (e.g. YOLOv8) to detect the **4 corners** of the ID card rather than just a standard bounding box.

| Condition | Behaviour |
|---|---|
| 4 corners found | Proceed to full `warpPerspective` |
| Fewer than 4 corners (Partial ID) | Fall back to standard bounding-box crop; tag metadata with `is_partial_document = True` |
| Face found but no ID elements | `label: selfie_instead_of_document` |

**Deskew & Flatten**

For full IDs, apply an OpenCV `warpPerspective` transformation using the 4 detected corners to remove background and normalize the shape.

**Orientation Correction**

Post-crop, pass the image through a lightweight **4-class orientation classifier** (0°, 90°, 180°, 270°) or check the text alignment vector from a quick low-res OCR pass. Apply `cv2.rotate` to ensure the document is right-side up before sending it downstream.

---

### Stage 2 — Presentation Attack Detection

Fraudsters attempt to trick the live-camera requirement by pointing their phone at a laptop screen displaying a stolen ID, or by photographing a printed paper forgery.

**Screen Replay Detection (Moiré)**

Train a CNN to detect **Moiré patterns** — the high-frequency, wavy visual distortions that naturally occur when a camera lens photographs a digital LCD/OLED screen.

**Print Attack / Material Texture**

Train the model to evaluate how the document interacts with light. Genuine plastic/polycarbonate IDs have distinct specular highlights (glare) and holographic properties.

- Uniform matte light absorption → `label: printed_paper_spoof`
- Raw paper edges → `label: printed_paper_spoof`

---

### Stage 3 — Camera Injection & Digital Tampering Detection

Advanced fraudsters use emulators or virtual cameras to bypass the live-photo UI, injecting a clean, digitally photoshopped image directly into the data stream.

**Error Level Analysis (ELA)**

Run ELA on the image. When an image is saved, JPEG compression is uniform. If a fraudster injected a file where text was digitally altered or pasted, the altered area will have a different compression signature. Train a CNN on ELA outputs to detect photoshopped text fields.

**Sensor Noise Analysis**

Live photos contain natural, uniform camera sensor noise. Injected images that have been digitally rendered or scrubbed often lack this noise pattern.

---

### Stage 4 — Template Verification & Font Anomalies

Ensure the physical card is an authentic Kenyan ID template, replacing brittle OCR keyword-matching logic.

**Layout Verification (Siamese Networks)**

Pass the deskewed ID through a **Siamese Network** to generate a dense vector embedding and calculate the distance against a "perfect" baseline template of a Kenyan ID.

> Note for Partial IDs: Skip or relax this threshold if `is_partial_document = True`, as a clipped card will skew the global layout embedding.

**Font Anomalies**

Train a model to examine bounding boxes from AWS OCR output. Evaluate whether the pixels and font weights around printed fields (e.g. "Name") appear heavily distorted, misaligned, or printed with a different ink density compared to other fields ("ID Number", "Date of Birth").

---

### Stage 5 — Orchestration & Tolerant Decision Matrix

To protect top-line conversion and satisfy business risk tolerances, the pipeline avoids aggressive automatic rejections. Instead it uses asynchronous shadow mode and risk-routing logic.

#### Phase 1 — Asynchronous Shadow Mode (0% conversion impact)

For the initial rollout (~30 days), all CV layers (YOLOv8, ELA, Moiré, Siamese Network) run **strictly in the background**.

- Models save scores and fraud classifications (`is_screen_replay`, `is_tampered`, `is_partial_document`) to database metadata but **do not block or alter the user journey**.
- **Goal:** Retroactively cross-reference model flags with actual first-payment defaults and collection data to build a concrete ROI case (e.g. "catches 60% of actual fraud rings while affecting only 1.5% of good users").

#### Phase 2 — Smart Risk Routing (Post-Shadow Activation)

Once the business approves model thresholds from Phase 1 data, the pipeline routes applications based on **risk tiers** rather than hard declines.

| Risk Tier | Condition | Action |
|---|---|---|
| High-Confidence Fraud Ring | High-probability digital tampering (ELA) or screen replay (Moiré) | Image passes to AWS OCR normally; application **silently routed to manual review queue** before payout |
| Garbage Photo | Zero ID components detected (blank wall, black screen, selfie in ID slot) | **Reupload triggered** |
| User Error / Partial Capture | `is_partial_document = True` or heavily rotated image | OpenCV auto-rotation layer salvages the image → AWS OCR; if Kenyan NIN is readable, application proceeds as normal |

---

## Metadata Flags

| Flag | Type | Description |
|---|---|---|
| `is_partial_document` | `bool` | True when fewer than 4 card corners were detected |
| `is_screen_replay` | `bool` | True when screen replay detected (Moiré / texture CNN) |
| `is_printout` | `bool` | True when print attack detected (v1: always False — deferred to v2) |
| `is_tampered` | `bool` | True when ELA or EXIF analysis detects digital manipulation |
| `id_type` | `str` | Detected document type: `legacy` / `maisha` / `huduma` / `passport` / `other_document` / `non_national_id` / `unknown` |
| `extracted_fields` | `dict` | OCR-extracted fields: `name`, `surname`, `sex`, `nationality`, `id_number`, `date_of_birth` (null if not extracted) |
| `field_extraction_confidence` | `float` | Average AWS Textract confidence (0–1) on extracted fields |
| `label` | `str` | Categorical outcome label (e.g. `selfie_instead_of_document`, `printed_paper_spoof`) |
| `risk_tier` | `str` | Decision matrix output: `high_fraud` / `garbage_photo` / `user_error` / `clean` |

---

## Models & Technologies

| Component | Technology |
|---|---|
| Corner / object detection | YOLOv8-OBB |
| Perspective correction | OpenCV `warpPerspective` |
| Orientation classification | 4-class heuristic (v1) → CNN (v2) |
| Screen replay detection | EfficientNet-B0 binary classifier |
| Print attack detection | Deferred to v2 |
| Tampering detection | ELA heuristic + EXIF analysis (v1) → CNN on ELA outputs (v2) |
| ID type classification | EfficientNet-B0 (v1: `legacy` vs `maisha` vs `other`) |
| Field extraction | AWS Textract + position-based parser per `id_type` |
| Template matching | Siamese Network (Phase 2) |
| OCR | AWS Textract / Rekognition |
