from doctr.io import DocumentFile
from doctr.models import ocr_predictor, detection_predictor
import json
import cv2
import math
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from utils import ProcessedPage


def convert_coordinates(geometry, page_dim):
    """
    Convert the coordinates from the OCR output where geom is % x/y values to the image coordinates
    """
    # % coordinate * width = pixel output
    len_x = page_dim[1]
    len_y = page_dim[0]
    (x_min, y_min) = geometry[0]
    (x_max, y_max) = geometry[1]
    x_min = math.floor(x_min * len_x)
    x_max = math.ceil(x_max * len_x)
    y_min = math.floor(y_min * len_y)
    y_max = math.ceil(y_max * len_y)
    return [x_min, x_max, y_min, y_max]

def edge_filter(image, geometry, page_width, page_height):
  """
  Filter out any text blocks near the edges of the page
  """
  page_padding_percentage = 0.025

  left_inner_x = math.floor(page_width * page_padding_percentage)
  right_inner_x = math.floor(page_width - (page_width * page_padding_percentage))
  top_inner_y = math.ceil(page_height * page_padding_percentage)
  bottom_inner_y = math.ceil(page_height - (page_height * page_padding_percentage))

  for geom in geometry:
    cv2.rectangle(image, (left_inner_x, top_inner_y), (right_inner_x, bottom_inner_y), (0,255,0), 3)
    if(geom["xmin"] >= left_inner_x and geom["xmax"] <= right_inner_x and geom["ymin"] >= top_inner_y and geom["ymax"] <= bottom_inner_y):
      cv2.rectangle(image, (geom["xmin"], geom["ymin"]), (geom["xmax"], geom["ymax"]), (255,0,0), 3)
    else:
      cv2.rectangle(image, (geom["xmin"], geom["ymin"]), (geom["xmax"], geom["ymax"]), (0,0,255), 3)
  
  return image

# def column_filter(image, geometry, page_width, page_height):
#    """
#    Filter out any text blocks that are not in columns
#    """
   

def load_images(dir):
    """
    Load all images in a directory
    """
    images = []
    dir_path = Path(dir)
    for image_path in dir_path.iterdir():
        if image_path.is_file() and image_path.suffix == '.jpg':
            img = cv2.imread(str(image_path))
            if img is not None:
                images.append({
                    "img": img,
                    "path": str(image_path),
                    "filename": image_path.name
                })
    return images

def draw_box_results(image, output):
  geoms = []
  for block in output["pages"][0]["blocks"]:
    coords = convert_coordinates(block["geometry"], output['pages'][0]["dimensions"])
    geoms.append({
      "xmin": coords[0],
      "xmax": coords[1],
      "ymin": coords[2],
      "ymax": coords[3],
      "area": (coords[1] - coords[0]) * (coords[3] - coords[2])
    })

  height, width = output["pages"][0]["dimensions"]
  output_dir = Path(output_path)
  output_file = output_dir / image["filename"]
  cv2.imwrite(str(output_file), edge_filter(cvimage, geoms, width, height))

def calculate_elbow_point(sum_of_squared_distances):
    # Convert the list to an array for easy array slicing
    distances = np.array(sum_of_squared_distances)

    # Get points for the line
    n_points = len(distances)
    all_coords = np.vstack((range(n_points), distances)).T

    # Get line between first and last point
    first_point = all_coords[0]
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))

    # Calculate the distance to the line for each point
    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_to_line = vec_from_first - np.outer(scalar_product, line_vec_norm)

    # Get the distance to the line
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))

    # The point with the maximum distance to the line is the elbow
    elbow_index = np.argmax(dist_to_line)

    return elbow_index

def optimal_clusters(x_values, max_clusters=10):
    X = np.array(x_values).reshape(-1, 1)
    sum_of_squared_distances = []

    for k in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=k).fit(X)
        sum_of_squared_distances.append(kmeans.inertia_)

    elbow_index = calculate_elbow_point(sum_of_squared_distances)
    print("Optimal number of clusters:", elbow_index + 1)

    plt.plot(range(1, max_clusters+1), sum_of_squared_distances, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distances')
    plt.title('Elbow Method For Optimal Clusters')
    plt.axvline(x=elbow_index + 1, color='r', linestyle='--')
    plt.show()

def find_clusters(x_values, num_clusters):
    X = np.array(x_values).reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    clusters = {i: [] for i in range(num_clusters)}

    for x, label in zip(x_values, kmeans.labels_):
      clusters[label].append(x)

    return clusters

def auto_canny(image, sigma=0.33):
  v = np.median(image)
  # apply automatic Canny edge detection using the computed median
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  edged = cv2.Canny(image, lower, upper)

  # return the edged image
  return edged

def preprocess_image(image, page_dimensions):
  
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(src = gray, ksize = (11, 11), sigmaX = 0)

  # high_thresh, thresh_im = cv2.threshold(blur, 0, 300, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,13,5)
  # cv2.imshow('thresh', thresh)
  # lowThresh = 0.5 * high_thresh

  # # Apply Canny edge detection on the grayscale image
  # edges = cv2.Canny(thresh, lowThresh, high_thresh, apertureSize=3)
  edges = auto_canny(thresh)
  edges =  cv2.dilate(edges, np.ones((5,5),dtype=np.uint8))
  edges =  cv2.erode(edges, np.ones((5,5),dtype=np.uint8))

  # # cv2.imshow('edges', edges)
  # # cv2.waitKey()
  # # cv2.destroyAllWindows()

  # # height, width = page_dimensions
  # # # cv2.imshow('image', edges)
  # # # cv2.waitKey()
  # # # cv2.destroyAllWindows()

  # # do this operation 5 times
  minLineLength = math.ceil(height * 0.1)
  lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=150, maxLineGap=5)

  if(lines is not None):
    # Display edges
    # cv2.imshow('edges', edges)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # Draw lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 5)

        if (abs(x1 - x2) < (width * 0.05)) and (abs(y1 - y2) > (height * 0.4)):  # Tolerance level for vertical lines
          cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 5)

    
  
  cv2.imshow('image', image)
  cv2.waitKey()
  cv2.destroyAllWindows()

def draw_columns(image, output):
  geoms = []
  for block in output["pages"][0]["blocks"]:
    for line in block["lines"]:
      coords = convert_coordinates(line["geometry"], output['pages'][0]["dimensions"])
      geoms.append({
        "xmin": coords[0],
        "xmax": coords[1],
        "ymin": coords[2],
        "ymax": coords[3],
        "area": (coords[1] - coords[0]) * (coords[3] - coords[2])
      })

  # # filter geoms into columns
  # xmins = [geom["xmin"] for geom in geoms]
  # # calculate the distribution of xmin
  # optimal_clusters(xmins)  # This will show the plot for the elbow method
  # # You can then decide the number of clusters based on the plot and find the clusters
  # num_clusters = 2  # Replace with your chosen number of clusters
  # clusters = find_clusters(xmins, num_clusters)
  # print(clusters)
     

  height, width = output["pages"][0]["dimensions"]
  output_dir = Path(output_path)
  output_file = output_dir / image["filename"]
  cv2.imwrite(str(output_file), edge_filter(cvimage, geoms, width, height))

def calculate_angle(line):
    x1, y1, x2, y2 = line
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

def extend_line(x1, y1, x2, y2, new_y):
    if x2 - x1 == 0:  # Avoid division by zero
        return x1, new_y
    slope = (y2 - y1) / (x2 - x1)
    new_x = int(x1 + (new_y - y1) / slope)
    return new_x, new_y

def line_detect(img, page_dimensions):
  height, width = page_dimensions

  scaling_factor = 0.25
  height = int(height * scaling_factor)
  width = int(width * scaling_factor)

  img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)
  thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,13,5)
  edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
  # edges =  cv2.dilate(edges, np.ones((9,9),dtype=np.uint8))
  # edges =  cv2.erode(edges, np.ones((9,9),dtype=np.uint8))

  lsd = cv2.createLineSegmentDetector(0)
  lines = lsd.detect(edges)[0]
  results = []
  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = map(int, line[0])
      # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

      if (abs(x1 - x2) < (width * 0.20)) and (abs(y1 - y2) > (height * 0.15)):  # Tolerance level for vertical lines
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
        results.append((x1, y1, x2, y2))

  # filter results to retrieve only x values that are 0 - (width / 2)
  left_lines = []
  right_lines = []

  for line in results:
    if line[0] <= (width / 2):
      left_lines.append(line)
    else:
       right_lines.append(line)
  
  # filter left_lines for the longest line
  longest_line = left_lines.sort(key=lambda line: line[3] - line[1], reverse=True)
  left_line = None
  if len(left_lines) > 0:
    longest_line = left_lines[0]
    x1, y1, x2, y2 = map(int, longest_line)
    x1, y1 = extend_line(x1, y1, x2, y2, 0)       # Top of the image
    x2, y2 = extend_line(x1, y1, x2, y2, height) # Bottom of the image
    left_line = (math.ceil(x1 / scaling_factor), math.ceil(y1 / scaling_factor), math.ceil(x2 / scaling_factor), math.ceil(y2 / scaling_factor))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
  
  # filter right_lines for the longest line
  longest_line = right_lines.sort(key=lambda line: line[3] - line[1], reverse=True)
  right_line = None
  if len(right_lines) > 0:
    longest_line = right_lines[0]
    x1, y1, x2, y2 = map(int, longest_line)
    x1, y1 = extend_line(x1, y1, x2, y2, 0)       # Top of the image
    x2, y2 = extend_line(x1, y1, x2, y2, height) # Bottom of the image
    
    right_line = (math.ceil(x1 / scaling_factor), math.ceil(y1 / scaling_factor), math.ceil(x2 / scaling_factor), math.ceil(y2 / scaling_factor))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)

  # cv2.imshow('Vertical Lines', img)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  
  return ProcessedPage(left_line, right_line)


image_path = "data"
output_path = "output"
model = ocr_predictor(pretrained=True)

for image in load_images("data"):
    doc = DocumentFile.from_images(image["path"])
    result = model(doc)
    output = result.export()

    cvimage = cv2.imread(image["path"])
    height, width = cvimage.shape[:2]

    # take the first 25% of the image and the last 25% of the image
    width_quarter = width // 2
    left_quarter = cvimage[:,0:math.ceil(width / 2)] # Left 25%
    right_quarter = cvimage[:,math.ceil(width / 2):width] # Right 25%
    # preprocess_image(left_quarter, output["pages"][0]["dimensions"])
    # preprocess_image(right_quarter, output["pages"][0]["dimensions"])
    # preprocess_image(cvimage, output["pages"][0]["dimensions"])
    # test_img(right_quarter, output["pages"][0]["dimensions"])
    # line_detect(left_quarter, output["pages"][0]["dimensions"])
    # line_detect(right_quarter, output["pages"][0]["dimensions"])
    img, left_line, right_line = line_detect(cvimage, output["pages"][0]["dimensions"])

    if left_line:
      cv2.line(cvimage, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 0), 5)
    if right_line:
      cv2.line(cvimage, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 0, 255), 5)
    
    cv2.imshow('Vertical Lines', cvimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # test_img(left_quarter, output["pages"][0]["dimensions"])

    # # Apply Otsu's thresholding on the grayscale image
    # high_thresh, thresh_im = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # lowThresh = 0.5 * high_thresh

    # # Apply Canny edge detection on the grayscale image
    # edges = cv2.Canny(blur, lowThresh, high_thresh)

    # height, width = output["pages"][0]["dimensions"]

    # minLineLength = height * 0.25
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=minLineLength, maxLineGap=10)

    # if(lines is not None):
    #   # Display edges
    #   cv2.imshow('edges', edges)
    #   cv2.waitKey()
    #   cv2.destroyAllWindows()
    #   # Draw lines on the image
    #   for line in lines:
    #       x1, y1, x2, y2 = line[0]
    #       cv2.line(cvimage, (x1, y1), (x2, y2), (0, 255, 0), 2)
      
    #   cv2.imshow('image', cvimage)
    #   cv2.waitKey()
    #   cv2.destroyAllWindows()
  # draw_columns(cvimage, output)
  # draw_box_results(cvimage, output)
  # geoms = []
  # for block in output["pages"][0]["blocks"]:
  #   for line in block["lines"]:
  #     coords = convert_coordinates(line["geometry"], output['pages'][0]["dimensions"])
  #     geoms.append({
  #       "xmin": coords[0],
  #       "xmax": coords[1],
  #       "ymin": coords[2],
  #       "ymax": coords[3],
  #       "area": (coords[1] - coords[0]) * (coords[3] - coords[2])
  #     })

  #   height, width = output["pages"][0]["dimensions"]
  #   output_dir = Path(output_path)
  #   output_file = output_dir / image["filename"]
  #   cv2.imwrite(str(output_file), edge_filter(cvimage, geoms, width, height))

# for geom in geoms:
  # if(geom["xmin"] >= xmin_lower_bound and geom["xmin"] <= xmin_upper_bound and geom["xmax"] >= xmax_lower_bound and geom["xmax"] <= xmax_upper_bound):
  #    cv2.rectangle(image, (geom["xmin"], geom["ymin"]), (geom["xmax"], geom["ymax"]), (255,0,0), 3)
  # else:
  #   cv2.rectangle(image, (geom["xmin"], geom["ymin"]), (geom["xmax"], geom["ymax"]), (0,0,255), 3)
  # if geom["area"] > avg_area:
  #   cv2.rectangle(image, (geom["xmin"], geom["ymin"]), (geom["xmax"], geom["ymax"]), (255,0,0), 3)
  # else:
  #   cv2.rectangle(image, (geom["xmin"], geom["ymin"]), (geom["xmax"], geom["ymax"]), (0,0,255), 3)


# xmin_values = [geom["xmin"] for geom in geoms]
# xminx_Q1, median, xmin_Q3 = np.percentile(xmin_values, [25, 50, 75])
# xmin_IQR = xmin_Q3 - xminx_Q1

# xmin_lower_bound = xminx_Q1 - 1.5 * xmin_IQR
# xmin_upper_bound = xmin_Q3 + 1.5 * xmin_IQR

# xmax_values = [geom["xmax"] for geom in geoms]
# xmaxx_Q1, median, xmax_Q3 = np.percentile(xmax_values, [25, 50, 75])
# xmax_IQR = xmax_Q3 - xmaxx_Q1
# xmax_lower_bound = xmaxx_Q1 - 1.5 * xmax_IQR
# xmax_upper_bound = xmax_Q3 + 1.5 * xmax_IQR

# # filter the geoms by average area
# avg_area = np.mean([geom["area"] for geom in geoms])

# for geom in geoms:
#   # if(geom["xmin"] >= xmin_lower_bound and geom["xmin"] <= xmin_upper_bound and geom["xmax"] >= xmax_lower_bound and geom["xmax"] <= xmax_upper_bound):
#   #    cv2.rectangle(image, (geom["xmin"], geom["ymin"]), (geom["xmax"], geom["ymax"]), (255,0,0), 3)
#   # else:
#   #   cv2.rectangle(image, (geom["xmin"], geom["ymin"]), (geom["xmax"], geom["ymax"]), (0,0,255), 3)
#   if geom["area"] > avg_area:
#     cv2.rectangle(image, (geom["xmin"], geom["ymin"]), (geom["xmax"], geom["ymax"]), (255,0,0), 3)
#   else:
#     cv2.rectangle(image, (geom["xmin"], geom["ymin"]), (geom["xmax"], geom["ymax"]), (0,0,255), 3)

# # result.show(doc)
# cv2.imshow('image', image)
# cv2.waitKey()
# cv2.destroyAllWindows()

# min x, min y
# left_x = output["pages"][0]["blocks"][0]["geometry"][0][0]
# right_y = output["pages"][0]["blocks"][0]["geometry"][0][1]

# result.show(doc)

# graphical_coordinates = get_coordinates(output)

# image = cv2.imread("data/single-column-numbered-3.jpg")

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# blur = cv2.GaussianBlur(src = gray, ksize = (3, 3), sigmaX = 0)

# _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# dilate = cv2.dilate(binary, kernel, iterations=4)

# cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     x,y,w,h = cv2.boundingRect(c)
#     cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)

# cv2.imshow('thresh', binary)
# cv2.imshow('dilate', dilate)
# cv2.imshow('image', image)
# cv2.waitKey()
# cv2.destroyAllWindows()
# cv2.imshow('Image with Bounding Box', binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# edges = cv2.Canny(binary, threshold1=30, threshold2=100)

# contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# largest_contour = max(contours, key=cv2.contourArea)
# x, y, w, h = cv2.boundingRect(largest_contour)

# cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
# cv2.imshow('Image with Bounding Box', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# model = ocr_predictor(pretrained=True)
# PDF
# doc = DocumentFile.from_images("data/index.jpg")
# Analyze
# result = model(doc)
# result.show(doc)
# doc = result.export()

# for each page in pages find the vertical text blocks
# for each vertical text block find the horizontal text blocks

# for page in pages:
#   print(page)
  

# word: (xmin, ymin), (xmax, ymax)


# with open("output.json", "w") as outfile: 
#     json.dump(doc, outfile)

# for page in doc["pages"]:
#   sentences = []
#   # Loop through each block
#   for block in page['blocks']:
#       for line in block['lines']:
#         words = line['words']
        
#         # Extract the value of each word and join them with a space to form a sentence
#         sentence = ' '.join(word['value'] for word in words)
        
#         # Add the sentence to the list of sentences
#         sentences.append(sentence)

# for sentence in sentences:
#     print(sentence)

