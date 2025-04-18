# Thread-Art

![Thread Art Generator](https://img.shields.io/badge/App-Thread%20Art%20Generator-blue)
![Python](https://img.shields.io/badge/Python-3.7+-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.15+-red)
![License](https://img.shields.io/badge/License-MIT-green)

A sophisticated web application that transforms ordinary images into stunning thread art representations using computational algorithms and digital simulation.

## Overview

The Thread Art Generator is an interactive application that creates string art simulations from user-uploaded images. It digitally replicates the traditional craft technique of stretching threads between a series of pins to form recognizable images. The application uses a greedy algorithm that strategically places lines between "nails" positioned around a circular frame to approximate the uploaded image.

## Features

- **Interactive Image Upload & Cropping**: Upload any image and crop it to focus on the subject
- **Customizable Parameters**: Adjust thread strength, maximum thread count, nail spacing, and output size
- **Live Generation Animation**: Watch the thread art being created in real-time
- **High-Quality Output**: Generate detailed thread art in your desired resolution
- **Downloadable Results**: Save your generated thread art as a PNG image
- **User-Friendly Interface**: Clean, intuitive design for easy navigation

## Installation

```bash
# Clone the repository
https://github.com/amanrjgit/Thread-Art.git
cd thread-art-generator

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- Streamlit
- NumPy
- Matplotlib
- scikit-image
- OpenCV (cv2)
- PIL
- streamlit-cropper
- streamlit-image-select

You can install all required packages using:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Upload an image (JPG, JPEG, or PNG format)

4. Crop the image if necessary to focus on the subject

5. Adjust the parameters:
   - **Thread Strength**: Controls how dark each thread appears
   - **Maximum Threads**: Sets the total number of threads to use
   - **Nail Spacing**: Adjusts the density of nails around the perimeter
   - **Output Size**: Determines the resolution of the final image

6. Click "Generate Thread Art" and watch the algorithm work

7. Download your finished thread art using the download button

## How It Works

The Thread Art Generator employs the following algorithm:

1. **Image Processing**: Converts the uploaded image to grayscale and normalizes it
2. **Nail Placement**: Positions virtual "nails" around a circular perimeter
3. **Thread Selection**: For each iteration:
   - Evaluates all possible connections from the current nail position
   - Selects the thread that best improves the approximation of the original image
   - Adds the selected thread to the canvas
   - Moves to the new nail position
4. **Visualization**: Renders the threads with anti-aliasing for a smooth appearance

The algorithm uses a greedy approach that maximizes image improvement with each new thread.

## Technical Implementation

- **Image Representation**: Images are processed as NumPy arrays
- **Thread Simulation**: Uses line anti-aliasing for realistic thread appearance
- **Optimization**: Implements efficient line drawing algorithms to speed up processing
- **UI**: Built with Streamlit for an interactive, responsive interface

## Performance Considerations

- Processing time depends on image size and thread count
- Higher thread counts produce more detailed results but require more processing time
- The application displays progress updates and statistics during generation

## Results

The Thread Art Generator produces impressive transformations from ordinary photographs to intricate thread art. Below are actual results from the application.

<div align="center">
  <img src="results/Screenshot 2025-04-18 223611.png" width="700px" alt="Thread Art Result 1"/>
  <p><em>Result with 1000 threads - Original image and thread art side by side</em></p>
  
  <img src="results/Screenshot 2025-04-18 224617.png" width="700px" alt="Thread Art Result 2"/>
  <p><em>Result with 2000 threads - Enhanced detail with more threads</em></p>
  
  <img src="results/Screenshot 2025-04-18 225359.png" width="700px" alt="Thread Art Result 3"/>
  <p><em>Result with 3000 threads - Maximum detail and definition</em></p>
</div>

Each image above shows the original photograph on the left and the corresponding thread art creation on the right. The application automatically displays these side-by-side comparisons so users can appreciate the transformation.

Notice how the thread art effectively captures the essence of the subject using only straight lines between points on a circle. The algorithm strategically places each thread to build up areas of varying darkness, creating a recognizable image that maintains the key characteristics of the original.

As demonstrated in the progression from 1000 to 3000 threads, increasing the thread count produces more refined results with greater detail and tonal variation. Even complex subjects with subtle gradients can be effectively represented through this technique.

These results showcase the Thread Art Generator's ability to transform portraits into unique artistic representations that combine mathematical precision with aesthetic appeal.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by traditional thread art techniques
- Built with [Streamlit](https://streamlit.io/) framework
- Image processing powered by [scikit-image](https://scikit-image.org/) and [OpenCV](https://opencv.org/)

---

Developed by Aman © 2025
