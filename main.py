from dotenv import load_dotenv
import os
import pytesseract
import cv2
import numpy as np
from PIL import Image

# Load .env file
load_dotenv()

# Set the tesseract command path from .env
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH')

def crop_to_page_and_preprocess(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Improved edge detection
    edged = cv2.Canny(blurred, 30, 200)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in image")

    # Assume the largest contour with four sides is the page
    page_contour = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:  # Looking for a rectangular contour
            page_contour = approx
            break

    if page_contour is None:
        raise ValueError("No rectangular contour found that could be the page")

    # Apply perspective transformation
    pts = page_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # Convert the warped image to grayscale, then threshold it
    warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(warp_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations
    kernel = np.ones((5,5), np.uint8)
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    return cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)


def detect_text_blocks(image_path, area_threshold=150, variance_threshold=10, max_area_ratio=0.5):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations
    kernel = np.ones((4,4), np.uint8)  # Adjusted kernel size
    dilate = cv2.dilate(thresh, kernel, iterations=1)  # Adjusted iterations

    # Find contours
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img.shape[0] * img.shape[1]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        if area < area_threshold or area > max_area_ratio * img_area:
            # Draw red boxes for filtered out contours
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            continue
        else:
            # Calculate standard deviation of the region
            roi = gray[y:y+h, x:x+w]
            std_dev = np.std(roi)
            if std_dev > variance_threshold:
                # Draw green boxes for contours that pass the filter
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur for noise reduction
    # blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Thresholding/Binarization
    # _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Deskewing or correcting page curl can be complex and may require custom implementation

    # Crop the image if necessary (you'll need to define the crop coordinates)
    # cropped = thresh[y:y+h, x:x+w]

    # Convert back to PIL Image
    final_image = Image.fromarray(gray)
    return final_image

def extract_text(image_path):
    """
    Extract text from image
    """
    try:
        text = pytesseract.image_to_string(preprocess_image(image_path), lang='eng', config='--psm 6')
        return text
    except Exception as e:
        print("Error:", e)

# cv2.imshow('Detected Text Blocks', crop_to_page_and_preprocess('data/table-of-contents.jpg'))
cv2.imshow('Detected Text Blocks', detect_text_blocks('data/index-test.jpg'))
cv2.waitKey(0)
cv2.destroyAllWindows()


# image = Image.open('data/testocr.png')
# try:
#     image = preprocess_image('data/table-of-contents.jpg')
#     # save image to file
#     image.save('data/boop.jpg')
#     print(extract_text('data/table-of-contents.jpg'))
# except Exception as e:
#     print("Error:", e)
