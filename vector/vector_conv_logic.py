import cv2
import numpy as np


def image_to_vector(image_path, output_path="output.svg"):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unable to load.")

    # Apply edge detection
    edges = cv2.Canny(img, 100, 200)

    # Extract coordinates of edges
    points = np.argwhere(edges > 0)
    # For simplicity, let's just output each point as a small line or circle
    # A more advanced approach would find contours and create polygonal approximations.

    # Create a basic SVG structure
    height, width = img.shape
    svg_header = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n'
    svg_footer = '</svg>'

    # Represent each edge point as a small circle in SVG
    svg_content = []
    for (y, x) in points:
        svg_content.append(f'<circle cx="{x}" cy="{y}" r="0.5" fill="black" />')

    svg_data = svg_header + "\n".join(svg_content) + "\n" + svg_footer

    with open(output_path, 'w') as f:
        f.write(svg_data)

    print(f"Vector representation saved to {output_path}")
