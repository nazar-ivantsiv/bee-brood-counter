import cv2
import numpy as np
import json

def generate_capped_brood_json(image_path, output_json_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color range for Capped Brood (Tan/Orange colors)
    # Adjust these values if lighting conditions vary
    lower_orange = np.array([10, 80, 100])   # Hue ~10-40 is typical for beeswax
    upper_orange = np.array([45, 255, 255])

    # Create a mask for capped cells
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Morphological operations to separate touching cells
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    
    # Filter contours to ensure we only get cells (ignore noise or frame edges)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Area filter: Cell size roughly 100-600 pixels (tweak based on resolution)
        if 80 < area < 800:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Append formatted coordinate
                points.append([cX, cY])

    # Structure for CSRNet/MSNN
    # Many implementations expect a simple list of points or a dictionary
    data = {
        "image": image_path,
        "count": len(points),
        "points": points  # List of [x, y]
    }

    # Save to JSON
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Successfully generated {len(points)} annotations in {output_json_path}")

# Run the function
generate_capped_brood_json(r'/Users/chip/claude_code_experiments/bee-brood-counter/bee_frame_sample.png', 'capped_brood_gt.json')