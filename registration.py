import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np


def load(image1_path, image2_path):
    '''
    '''
    image1 = cv.imread(image1_path)
    if image1 is None:
      raise FileNotFoundError("Image 1 not found")
        
    image2 = cv.imread(image2_path)

    if image2 is None:
      raise FileNotFoundError("Image 2 not found")

    return image1, image2 


def display_images(image1,image2,img1_day,img2_day):

    image1_rgb = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
    image2_rgb = cv.cvtColor(image2, cv.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image1_rgb)
    axes[0].set_title(f"Day {img1_day}")
    axes[0].axis("off")
    axes[1].imshow(image2_rgb)
    axes[1].set_title(f"Day {img2_day}")
    axes[1].axis("off")

    plt.tight_layout(pad=2.0)
    plt.show()
   
def process_image(image):

    # scale image down by a factor of 1/3
    image = cv.resize(image,(4500,3000),interpolation=cv.INTER_AREA)

    # convert to grayscale
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    
    return image, image.shape

def mask_image(image, model,processor, device,image_dimensions):
    '''
    '''
    image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    # this is location of what we want to segment
    # for now it's at the center
    input_points = [[[image_dimensions[1]/2, image_dimensions[0]/2]]]

    inputs = processor(image, input_points=input_points, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # returns list of binary masks
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )
    #print(masks[0])
  
    best_mask_index = outputs.iou_scores.argmax().item()
    print(best_mask_index)
    
    # Convert the best mask (highest IoU) to a 1-channel uint8 array (0 or 255) for OpenCV
    mask = masks[0][0][best_mask_index].numpy().astype(np.uint8) * 255

    return mask


def detect_features(image, mask, feature_detection_type,max_keypoints=None):
   
   if feature_detection_type == "ORB":
      keypoints, descriptors = orb_feature_detection(max_keypoints,image,mask)
      return keypoints, descriptors
   
   elif feature_detection_type == "SIFT":
      keypoints, descriptors =  sift_feature_detection(max_keypoints, image,mask)
      return keypoints,descriptors
   
   elif feature_detection_type == "AKAZE":
      keypoints, descriptors = akaze_feature_detection(image,mask)
      return keypoints, descriptors
   else:
      raise ValueError("Detection type not found")


def akaze_feature_detection(image,mask):
   akaze = cv.AKAZE.create()
   keypoints, descriptors = akaze.detectAndCompute(image,mask)

   return keypoints, descriptors

def orb_feature_detection(max_keypoints, image, mask):

    orb = cv.ORB.create(max_keypoints)
    keypoints, descriptors = orb.detectAndCompute(image, mask)

    return keypoints, descriptors

def sift_feature_detection(max_keypoints,image,mask):

    sift = cv.SIFT.create(max_keypoints)
    keypoints,descriptors = sift.detectAndCompute(image,mask)

    return keypoints,descriptors

def FlannMatcher(kpsA, descsA, kpsB, descsB, feature):
    if feature == "SIFT":
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
    else:
        # LSH index for binary descriptors (ORB, AKAZE)
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,
            key_size=12,
            multi_probe_level=1
        )
        search_params = dict()

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descsA, descsB, k=2)

    good = []
    for m in matches:
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            good.append(m[0])
    good = sorted(good, key=lambda x: x.distance)

    ptsA = np.float32([kpsA[m.queryIdx].pt for m in good[:10]]).reshape(-1, 1, 2)
    ptsB = np.float32([kpsB[m.trainIdx].pt for m in good[:10]]).reshape(-1, 1, 2)
    return ptsA, ptsB, good


def BfMatcher(kpsA, descsA, kpsB, descsB, feature):
    if feature == "SIFT":
        norm = cv.NORM_L2          # L2 is more accurate than L1 for SIFT
    else:
        norm = cv.NORM_HAMMING     # required for binary descriptors (ORB, AKAZE)

    # crossCheck=True: a match is only kept if it's the best match in BOTH directions
    # This replaces the ratio test — don't use knnMatch here
    matcher = cv.BFMatcher(norm, crossCheck=True)
    matches = matcher.match(descsA, descsB)
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:10]

    ptsA = np.float32([kpsA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsB = np.float32([kpsB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return ptsA, ptsB, good


def match_features(kpsA, descsA, kpsB, descsB, feature_detection_type, matcher_type="BF"):
    '''
    Dispatches to the correct matcher.

    matcher_type : "BF" or "FLANN"
    feature_detection_type : "SIFT", "ORB", or "AKAZE"

    Returns: ptsA, ptsB, good_matches
    '''
    if matcher_type == "BF":
        return BfMatcher(kpsA, descsA, kpsB, descsB, feature_detection_type)
    elif matcher_type == "FLANN":
        return FlannMatcher(kpsA, descsA, kpsB, descsB, feature_detection_type)
    else:
        raise ValueError(f"Unknown matcher_type '{matcher_type}'. Choose 'BF' or 'FLANN'.")

def visualize_matches(image_1, image_2, keypoints_1, keypoints_2, matches):
    '''
    '''
    image = cv.drawMatches(image_1,keypoints_1,image_2, keypoints_2, matches, None, flags= cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(10,5))
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def affine_transform(image_1_pts,image_2_pts,image_1,image_2 ):
  
  (M, inliers) = cv.estimateAffine2D(image_2_pts,image_1_pts, cv.RANSAC)
  print(len(inliers))
  if M is None:
    print("Tranformation matrix not found")
    return None

  (h, w) = image_1.shape[:2]
  aligned_image = cv.warpAffine(image_2, M, (w, h))
  

  return aligned_image

def homography(image_1_pts,image_2_pts, image_1,image_2):
  (H, mask) = cv.findHomography(image_2_pts, image_1_pts,cv.RANSAC)
  if H is None:
    print("Transformation matrix not found")
    return None

  (h, w) = image_1.shape[:2]
  aligned_image = cv.warpPerspective(image_2, H, (w, h))
  

  return aligned_image

