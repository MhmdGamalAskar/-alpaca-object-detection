"""


import streamlit as st
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from PIL import Image
import numpy as np
from training.util import visualize, set_background

set_background('./bg3.png')


st.title("🦙 Alpaca Object Detection")
st.header("Upload an image to detect alpacas")


file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

cfg.MODEL.WEIGHTS = "./model_final.pth"
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)

if file :
    image = Image.open(file)
    image_array = np.asarray(image)

    outputs = predictor(image_array)


    preds = outputs["instances"].pred_classes.tolist()
    scores = outputs["instances"].scores.tolist()
    bboxes = outputs["instances"].pred_boxes

    threshold = 0.5
    bboxes_ = []
    scores_ = []

    for j, bbox in enumerate(bboxes):

        bbox = bbox.tolist()
        score = scores[j]

        if score > threshold:
            x1, y1, x2, y2 = [int(i) for i in bbox]

            bboxes_.append([x1, y1, x2, y2])
            scores_.append(score)
    # messages
    if bboxes_:

        n = len(bboxes_)

        if n == 1:
            st.success("🦙 1 alpaca detected in the image.")
        else:
            st.success(f"🦙 {n} alpacas detected in the image.")

    else:
        st.info("No alpacas were detected in this image.")

    # visualize
    visualize(image, bboxes_, scores_)

"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

import streamlit as st
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from PIL import Image
import numpy as np
from training.util import visualize, set_background

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alpaca Detector",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)


bg_path = BASE_DIR / "bg3.png"

set_background(str(bg_path))

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Card-style containers */
    .metric-card {
        background: rgba(255,255,255,0.12);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card h1 { font-size: 2.5rem; margin: 0; }
    .metric-card p  { margin: 0; opacity: 0.8; font-size: 0.9rem; }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(255,255,255,0.4);
        border-radius: 16px;
        padding: 1rem;
        background: rgba(255,255,255,0.08);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15,15,25,0.75);
        backdrop-filter: blur(14px);
    }

    /* Title */
    h1 { text-shadow: 0 2px 12px rgba(0,0,0,0.5); }
</style>
""", unsafe_allow_html=True)


# ── Cached model loader ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(score_thresh: float):
    """Load the DetectronPredictor once and cache it."""
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.WEIGHTS = "./models/model_final.pth"
    cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1, max_value=0.95,
        value=0.50, step=0.05,
        help="Only detections above this score will be shown."
    )

    st.markdown("---")
    st.markdown("## 🧠 Model Info")
    st.markdown("""
    | | |
    |---|---|
    | **Architecture** | RetinaNet |
    | **Backbone** | ResNet-101 FPN |
    | **Pretrained** | COCO |
    | **AP @ 0.50** | 92.0 % |
    | **AP @ 0.50:0.95** | 74.6 % |
    """)

    st.markdown("---")
    st.markdown("## 📖 How to use")
    st.markdown("""
    1. Upload a **JPG / PNG** image  
    2. Adjust the **confidence threshold** if needed  
    3. View the detected alpacas with bounding boxes  
    """)

    st.markdown("---")
    st.caption("Built with Detectron2 · Streamlit · Plotly")


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🦙 Alpaca Object Detection")
st.markdown("##### Upload an image and the model will locate every alpaca in it.")
st.markdown("---")


# ── Load model ─────────────────────────────────────────────────────────────────
with st.spinner("🔄 Loading model weights …"):
    try:
        predictor = load_model(threshold)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()


# ── File uploader ──────────────────────────────────────────────────────────────
file = st.file_uploader(
    "Choose an image",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed"
)

if file:
    # ── Validate & convert image ───────────────────────────────────────────────
    try:
        image = Image.open(file).convert("RGB")
        image_array = np.asarray(image)
    except Exception:
        st.error("❌ Could not read the uploaded file. Please try a different image.")
        st.stop()

    # ── Run inference ──────────────────────────────────────────────────────────
    with st.spinner("🔍 Detecting alpacas …"):
        try:
            outputs = predictor(image_array)
        except Exception as e:
            st.error(f"❌ Inference failed: {e}")
            st.stop()

    # ── Parse results ──────────────────────────────────────────────────────────
    scores = outputs["instances"].scores.tolist()
    bboxes_raw = outputs["instances"].pred_boxes

    bboxes_, scores_ = [], []
    for j, bbox in enumerate(bboxes_raw):
        score = scores[j]
        if score >= threshold:
            x1, y1, x2, y2 = [int(v) for v in bbox.tolist()]
            bboxes_.append([x1, y1, x2, y2])
            scores_.append(score)

    # ── Metrics row ────────────────────────────────────────────────────────────
    n = len(bboxes_)
    avg_conf = sum(scores_) / n if n else 0.0
    max_conf = max(scores_) if n else 0.0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h1>{'🦙 ' * min(n, 5) if n else '—'}</h1>
            <p><b>{n}</b> alpaca{'s' if n != 1 else ''} detected</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h1>{avg_conf:.0%}</h1>
            <p>Average confidence</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h1>{max_conf:.0%}</h1>
            <p>Top confidence</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Status message ─────────────────────────────────────────────────────────
    if bboxes_:
        st.success(f"✅ Found {n} alpaca{'s' if n != 1 else ''} — see bounding boxes below.")
    else:
        st.info("🔍 No alpacas detected above the current threshold. Try lowering it in the sidebar.")

    # ── Visualize ──────────────────────────────────────────────────────────────
    visualize(image, bboxes_, scores_)

    # ── Raw detections expander ────────────────────────────────────────────────
    if bboxes_:
        with st.expander("📋 Raw detection data"):
            for i, (bbox, score) in enumerate(zip(bboxes_, scores_), 1):
                st.markdown(
                    f"**Detection {i}** — confidence: `{score:.4f}` | "
                    f"bbox: `[x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}]`"
                )

else:
    # ── Placeholder when no image is uploaded ─────────────────────────────────
    st.markdown("""
    <div style="
        text-align:center;
        padding: 4rem 2rem;
        background: rgba(255,255,255,0.07);
        border-radius: 20px;
        border: 2px dashed rgba(255,255,255,0.2);
        margin-top: 1rem;
    ">
        <div style="font-size:5rem">🦙</div>
        <h3 style="margin-top:1rem">No image uploaded yet</h3>
        <p style="opacity:0.7">Upload a JPG or PNG above to get started</p>
    </div>
    """, unsafe_allow_html=True)