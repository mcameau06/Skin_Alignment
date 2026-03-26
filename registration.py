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
    combined = cv.hconcat([image1_rgb,image2_rgb])
    
    plt.imshow(combined)
    plt.axis("off")
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
    

def match_features(descriptors_1,descriptors_2, feature_detection_type):
    '''
    '''
    
    if feature_detection_type == "ORB":
        method = cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    
    elif feature_detection_type == "SIFT":
        method = cv.DESCRIPTOR_MATCHER_BRUTEFORCE_SL2

    else:
        raise ValueError("Invalid feature detection type. Choose 'ORB' or 'SIFT'")
        

    matcher = cv.DescriptorMatcher.create(method)
    matches = matcher.match(descriptors_1,descriptors_2)
    all_matches = sorted(matches, key=lambda x: x.distance)
    all_matches = all_matches[:100]

    
    return all_matches

def orb_feature_detection(max_keypoints, image, mask):

    orb = cv.ORB.create(max_keypoints)
    keypoints, descriptors = orb.detectAndCompute(image, mask)

    return keypoints, descriptors

def sift_feature_detection(max_keypoints,image,mask):

    sift = cv.SIFT.create(max_keypoints)
    keypoints,descriptors = sift.detectAndCompute(image,mask)

    return keypoints,descriptors

def visualize_matches(image_1, image_2, keypoints_1, keypoints_2, matches):
    '''
    '''
    img3 = cv.drawMatches(image_1,keypoints_1,image_2, keypoints_2, matches, None, flags= cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img3 = cv.cvtColor(img3, cv.COLOR_BGR2RGB)

    plt.imshow(img3)
    plt.axis("off")
    plt.show()

