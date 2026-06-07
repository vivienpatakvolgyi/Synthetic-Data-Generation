import os
import cv2
import numpy as np
import glob

# Settings

OUTPUT_DIR = "lum_crop"

MIN_AREA = 20
MAX_AREA = 5000

MIN_CIRCULARITY = 0.60
MIN_SOLIDITY = 0.80

EXTRA_MARGIN = 15
MIN_FILL_RATIO = 0.6
AURA_PIXELS = 15


# Helper functions
def contour_circularity(contour):
    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return 0

    return 4 * np.pi * area / (perimeter * perimeter)


def contour_solidity(contour):
    area = cv2.contourArea(contour)

    hull = cv2.convexHull(contour)

    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        return 0

    return area / hull_area


# Output dir

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all _lum.jpg files in current directory
image_files = glob.glob("*_lum.jpg")

print(f"Found {len(image_files)} _lum.jpg files to process")

# Process each image
for IMAGE_PATH in image_files:
    print(f"\nProcessing: {IMAGE_PATH}")
    
    # Load images

    img = cv2.imread(IMAGE_PATH)

    if img is None:
        print(f"  Failed to load image: {IMAGE_PATH}")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    green = img[:, :, 1] # green channel

    # Preprocessing
    blur = cv2.GaussianBlur(green, (5, 5), 0)

    binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    kernel = np.ones((3, 3), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # Contours

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"  Number of objects: {len(contours)}")

    saved = 0


    img_contours = img.copy()

    cv2.drawContours(
        img_contours,
        contours,
        -1,
        (0, 0, 255),
        1
    )

    # Process contours

    for contour in contours:

        # AREA

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        if area > MAX_AREA:
            continue

        # CIRCULARITY

        circularity = contour_circularity(contour)

        if circularity < MIN_CIRCULARITY:
            continue

        # SOLIDITY

        solidity = contour_solidity(contour)

        if solidity < MIN_SOLIDITY:
            continue

        # FILL RATIO

        (cx_f, cy_f), radius_f = cv2.minEnclosingCircle(contour)

        circle_area = np.pi * radius_f * radius_f

        if circle_area == 0:
            continue

        fill_ratio = area / circle_area

        if fill_ratio < MIN_FILL_RATIO:
            continue

        # CROP SIZE

        cx = int(cx_f)
        cy = int(cy_f)
        radius = int(radius_f)

        crop_radius = radius + EXTRA_MARGIN

        img_h, img_w = img.shape[:2]

        x1 = max(0, cx - crop_radius)
        y1 = max(0, cy - crop_radius)

        x2 = min(img_w, cx + crop_radius)
        y2 = min(img_h, cy + crop_radius)

        # FILTERING OUT MULTIPLE OBJECTS

        green_crop = green[y1:y2, x1:x2]

        bright = cv2.threshold(
            green_crop,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            bright,
            connectivity=8
        )

        large_objects = 0

        for i in range(1, num_labels):

            component_area = stats[i, cv2.CC_STAT_AREA]

            if component_area > 20:
                large_objects += 1

        if large_objects > 1:
            continue

        # ORIGINAL CELL MASK

        cell_mask = np.zeros(
            (img_h, img_w),
            dtype=np.uint8
        )

        cv2.drawContours(
            cell_mask,
            [contour],
            -1,
            255,
            thickness=-1
        )

        # AURA

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                2 * AURA_PIXELS + 1,
                2 * AURA_PIXELS + 1
            )
        )

        aura_mask = cv2.dilate(
            cell_mask,
            kernel
        )

        # SOFT ALPHA
        distance = cv2.distanceTransform(
            aura_mask,
            cv2.DIST_L2,
            5
        )

        if distance.max() == 0:
            continue

        distance = distance / distance.max()

        # transition character
        distance = np.power(
            distance,
            2.0
        )

        alpha = (
            distance * 255
        ).astype(np.uint8)

        # the cell itself is fully opaque
        alpha[cell_mask > 0] = 255

        # RGBA

        rgba = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2BGRA
        )

        rgba[:, :, 3] = alpha

        # CROP

        crop = rgba[
            y1:y2,
            x1:x2
        ]


        # SAVE

        out_name = os.path.join(
            OUTPUT_DIR,
            f"cell_{IMAGE_PATH.split('.')[0]}_{saved:05d}.png"
        )

        cv2.imwrite(
            out_name,
            crop
        )

        saved += 1

    print(f"  Saved cells: {saved}")

print(f"\nProcessing complete!")