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
    
    image1 = cv.resize(image1,(500,500))
    
    image2 = cv.imread(image2_path)

    if image2 is None:
      raise FileNotFoundError("Image 2 not found")

    image2 = cv.resize(image2, (500,500))

    return image1, image2 


def display_images(image1,image2):

    image1_rgb = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
    image2_rgb = cv.cvtColor(image2, cv.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image1_rgb)
    axes[0].axis("off")
    axes[1].imshow(image2_rgb)
    axes[1].axis("off")

    plt.tight_layout(pad=2.0)
    plt.show()
    


def mask_image(image, model,processor, device):
    '''
    '''
    image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    # this is location of what we want to segment
    # for now it's at the center
    input_points = [[[250, 250]]]

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
    
    # Convert the best mask to a 1-channel uint8 array (0 or 255) for OpenCV
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

def match_features(descriptors_1,descriptors_2,image_1_keypoints, image_2_keypoints, feature_detection_type):
    '''
    returns 
    matches of image 1, matches of image 2, descriptor matcher object
    '''
    
    if feature_detection_type == "ORB":
        method = cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    
    elif feature_detection_type == "SIFT":
        method = cv.DESCRIPTOR_MATCHER_BRUTEFORCE_SL2
    elif feature_detection_type == "AKAZE":
       method = cv.DescriptorMatcher_BRUTEFORCE_HAMMING

    else:
        raise ValueError("Invalid feature detection type. Choose 'ORB' or 'SIFT'")
        

    matcher = cv.DescriptorMatcher.create(method)
    matches = matcher.knnMatch(descriptors_1, descriptors_2, k=2)

    best_matches = []
    for match_pair in matches:
      if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < .75 * n.distance:
          best_matches.append(m)

    best_matches = sorted(best_matches, key=lambda x: x.distance)
    
    # extract best matches
    ptsA = np.array([image_1_keypoints[m.queryIdx].pt for m in best_matches], dtype="float32").reshape(-1, 1, 2)
    ptsB = np.array([image_2_keypoints[m.trainIdx].pt for m in best_matches], dtype="float32").reshape(-1, 1, 2)

    return ptsA, ptsB,best_matches


def visualize_matches(image_1, image_2, keypoints_1, keypoints_2, matches):
    '''
    '''
    image = cv.drawMatches(image_1,keypoints_1,image_2, keypoints_2, matches, None, flags= cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

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

