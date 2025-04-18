import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.draw import line, circle_perimeter, line_aa
from skimage.transform import resize
from skimage import io
from math import atan2
import time
import cv2
from PIL import Image, ImageOps
import io as python_io
from streamlit_cropper import st_cropper
import base64
from streamlit_image_select import image_select

st.set_page_config(
    page_title="String Art Generator",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #7f8c8d;
        font-size: 0.8em;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #3498db;
        margin-bottom: 20px;
    }
    .stProgress > div > div > div > div {
        background-color: #3498db;
    }
    </style>
    """, unsafe_allow_html=True)


# Helper functions from your provided code
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def largest_square(image: np.ndarray) -> np.ndarray:
    short_edge = np.argmin(image.shape[:2])  # 0 = vertical <= horizontal; 1 = otherwise
    short_edge_half = image.shape[short_edge] // 2
    long_edge_center = image.shape[1 - short_edge] // 2
    if short_edge == 0:
        return image[:, long_edge_center - short_edge_half:
                        long_edge_center + short_edge_half]
    if short_edge == 1:
        return image[long_edge_center - short_edge_half:
                     long_edge_center + short_edge_half, :]


def white_canvas(picture):
    height = len(picture)
    width = len(picture[0])
    return np.ones((height, width))

def black_canvas(picture):
    height = len(picture)
    width = len(picture[0])
    return np.zeros((height, width))

def create_circle_nail_positions(picture, nail_step=2):
    height = len(picture)
    width = len(picture[0])

    centre = (height // 2, width // 2)
    radius = min(height, width) // 2 - 1
    rr, cc = circle_perimeter(centre[0], centre[1], radius)
    nails = list(set([(rr[i], cc[i]) for i in range(len(cc))]))
    nails.sort(key=lambda c: atan2(c[0] - centre[0], c[1] - centre[1]))
    nails = nails[::nail_step]

    return nails

def get_aa_line(from_pos, to_pos, str_strength, picture):
    rr, cc, val = line_aa(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
    # build up string density rather than subtract
    line = picture[rr, cc] + str_strength * val
    line = np.clip(line, 0, 1)
    return line, rr, cc

def find_best_nail_position(current_position, nails, str_pic, orig_pic, str_strength):
    best_cumulative_improvement = -np.inf
    best_nail_position = None
    best_nail_idx = None

    for nail_idx, nail_position in enumerate(nails):
        # skip drawing back to the same nail
        if nail_position == current_position:
            continue

        overlayed_line, rr, cc = get_aa_line(current_position, nail_position, str_strength, str_pic)
        before = (str_pic[rr, cc] - orig_pic[rr, cc])**2
        after  = (overlayed_line    - orig_pic[rr, cc])**2
        cumulative_improvement = np.sum(before - after)

        if cumulative_improvement >= best_cumulative_improvement:
            best_cumulative_improvement = cumulative_improvement
            best_nail_position = nail_position
            best_nail_idx = nail_idx

    return best_nail_idx, best_nail_position, best_cumulative_improvement

# Function to convert image to displayable format (uint8)
def prepare_for_display(img):
    # Normalize to 0-1 range if not already
    if img.dtype != np.uint8:
        img_normalized = img.copy()
        if img.max() > 1.0 or img.min() < 0.0:
            img_normalized = (img - img.min()) / (img.max() - img.min())
        # Convert to uint8 for display
        img_display = (img_normalized * 255).astype(np.uint8)
    else:
        img_display = img
    return img_display

# Function to draw nails on image
def draw_nails(img, nail_positions):
    img_with_nails = prepare_for_display(img).copy()
    if len(img_with_nails.shape) == 2:  # Grayscale
        img_with_nails = cv2.cvtColor(img_with_nails, cv2.COLOR_GRAY2BGR)

    for nail in nail_positions:
        y, x = nail
        cv2.circle(img_with_nails, (int(x), int(y)), 2, (0, 0, 0), -1)
    return img_with_nails


def get_image_download_link(img, filename, text):
    """Generate a download link for an image"""
    buffered = python_io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}" class="download-button">{text}</a>'
    return href


# Update your generate_string_art_with_animation function to better handle line visualization
def generate_string_art_with_animation(orig_pic, str_pic, nails, str_strength, max_iterations, animation_placeholder,
                                       status_text, progress_bar):

    # Initialize variables
    current_position = nails[0]
    prev_position = current_position
    pull_order = [0]
    current_str_pic = str_pic.copy()

    # Create timing variables
    start_time = time.time()
    iter_times = []

    # Main loop
    for i in range(max_iterations):
        # Update progress
        if i % 10 == 0:
            progress_percentage = (i + 1) / max_iterations
            progress_bar.progress(progress_percentage)
            status_text.text(f"Processing: {i + 1}/{max_iterations} strings drawn ({progress_percentage * 100:.1f}%)")

        # Measure iteration time
        start_iter = time.time()

        # Find best nail
        idx, best_nail_position, best_cumulative_improvement = find_best_nail_position(
            current_position, nails, current_str_pic, orig_pic, str_strength)
        pull_order.append(idx)


        # Apply line to image - THIS IS THE CRITICAL PART
        best_overlayed_line, rr, cc = get_aa_line(current_position, best_nail_position, str_strength, current_str_pic)
        current_str_pic[rr, cc] = best_overlayed_line  # Update the image with the line

        # Update current position
        current_position = best_nail_position

        # Track iteration time
        iter_time = time.time() - start_iter
        iter_times.append(iter_time)

        # Update animation frame periodically
        if i % 50 == 0 or i == max_iterations - 1:
            # Prepare image for display
            display_img = prepare_for_display(current_str_pic)
            display_img_rgb = (cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                               if display_img.ndim == 2
                               else display_img.copy())

            # Draw nails in light grey, anti‑aliased
            for y, x in nails:
                cv2.circle(
                    display_img_rgb,
                    (int(x), int(y)),
                    2,
                    (200, 200, 200),
                    -1,
                    lineType=cv2.LINE_AA
                )

            # Draw the most recent string in a soft grey, anti‑aliased
            # (prev_position, current_position updated above)
            y0, x0 = prev_position
            y1, x1 = current_position
            cv2.line(
                display_img_rgb,
                (int(x0), int(y0)),
                (int(x1), int(y1)),
                (0, 0, 0),
                1,
                lineType=cv2.LINE_AA
            )

            animation_placeholder.image(
                ImageOps.invert(Image.fromarray(display_img_rgb)),
                caption="String Art Progress",
                use_container_width=True
            )

    elapsed = time.time() - start_time
    avg_time_per_iter = np.mean(iter_times) * 1000 if iter_times else 0

    # Return the final state and statistics
    return current_str_pic, elapsed, avg_time_per_iter

st.title("🧵 Thread Art Generator")

with st.expander("About Thread Art", expanded=False):
    st.markdown("""
    ## What is Thread Art?

    Thread art is a technique for creating images by stringing thread between pins arranged in a pattern, 
    typically on a board. This app simulates the process digitally, creating an image made of straight lines 
    that, when viewed together, form a recognizable image.

    ## How it Works

    1. Upload a portrait image
    2. Crop it to focus on the subject if needed
    3. Adjust the settings like Thread strength and maximum Threads
    4. Generate your Thread art
    5. Download the result

    The algorithm places "nails" around a circle and then connects these nails with Threads (straight lines).
    Each Thread is chosen to make the image look more like the original.
    """)

# Sidebar for instructions and parameters
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    <div class="info-box">
    1. Upload a portrait photo
    2. Make sure the subject is centered in the image
    3. Crop the image if needed
    4. Adjust the parameters
    5. Generate your Thread art
    6. Download the result
    </div>
    """, unsafe_allow_html=True)

    st.header("Parameters")
    string_strength = st.slider("Thread Strength", min_value=0.1, max_value=1.0, value=0.3, step=0.05,
                                help="Controls how dark each Thread appears. Higher values create darker Threads.")

    max_iterations = st.slider("Maximum Threads", min_value=500, max_value=5000, value=1000, step=100,
                               help="Maximum number of Threads to draw. More Threads give more detail but take longer.")

    nail_step = st.slider("Nail Spacing", min_value=1, max_value=10, value=4, step=1,
                          help="Controls the spacing between nails. Lower values mean more nails and potentially more detail.")

    image_size = st.slider("Output Size", min_value=200, max_value=800, value=600, step=50,
                           help="Size of the output image in pixels.")

# Main content
# Single column for image upload and cropping
uploaded_file = st.file_uploader("Upload a portrait image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)

    # Display guidance for cropping
    st.info("Ensure your subject is centered in the image. Use the cropping tool to focus on your subject if needed.")

    # Display cropping interface in single column
    box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
    aspect_ratio = (1, 1)
    cropped_img = st_cropper(image, box_color=box_color, aspect_ratio=aspect_ratio)

    st.caption("Drag to crop and ensure your subject is centered")

    # Preprocess image button
    if st.button("Generate Thread Art"):
        # Convert PIL Image to numpy array
        img_array = np.array(cropped_img)

        # Convert to grayscale if needed
        if len(img_array.shape) > 2:
            img_array = rgb2gray(img_array)

        # Make square and resize
        img_array = largest_square(img_array)
        img_array = resize(img_array, (image_size, image_size))
        img_array_norm = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)

        # Set up placeholders for live updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        animation_placeholder = st.empty()
        nails = create_circle_nail_positions(img_array_norm, nail_step)
        orig_pic = 1 - img_array_norm
        str_pic = black_canvas(img_array)

        # Generate Thread art with animation
        with st.spinner("Processing... Please wait."):
            result_pic, elapsed_time, avg_iter_time = generate_string_art_with_animation(
                orig_pic, str_pic, nails,string_strength, max_iterations,
                animation_placeholder, status_text, progress_bar
            )

        # Display completion status
        st.success(f"✅ Thread art generated in {elapsed_time:.2f} seconds!")

        # Create columns for results
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(img_array_norm, use_container_width=True)
            st.text(f"Size: {image_size}x{image_size} pixels")

        with col2:
            st.subheader("Thread Art Result")
            # Convert numpy array to PIL Image for display and download
            result_img = ImageOps.invert(Image.fromarray((result_pic * 255).astype(np.uint8)))
            result_img_np = np.array(result_img).astype(np.float32)
            result_img_norm = (result_img_np - result_img_np.min()) / (result_img_np.max() - result_img_np.min() + 1e-8)
            st.image(result_img_norm, use_container_width=True)

            # Download button
            st.markdown(get_image_download_link(result_img, "Thread_art.png", "📥 Download Thread Art"),
                        unsafe_allow_html=True)

        # Display statistics
        st.subheader("Statistics")
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1:
            st.metric("Total Threads", f"{max_iterations}")
        with stats_col2:
            st.metric("Processing Time", f"{elapsed_time:.2f}s")
        with stats_col3:
            st.metric("Nails Used", f"{len(nails)}")

        st.caption(f"Average time per Thread: {avg_iter_time:.2f}ms")

# Footer
st.markdown("""
<div class="footer">
Thread Art Generator © 2025 | Created with Streamlit | Aman
</div>
""", unsafe_allow_html=True)