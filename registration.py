import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from numpy._core.multiarray import interp
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

                        # image displaying

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


def visualize_matches(image_1, image_2, keypoints_1, keypoints_2, matches):
    '''
    '''
    image = cv.drawMatches(image_1,keypoints_1,image_2, keypoints_2, matches, None, flags= cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # convert to rgb for matplotlib
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(10,5))
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def visualize_keypoints(image_1,image_2, keypoints_1, keypoints_2):

    output_image_1 = cv.drawKeypoints(image_1, keypoints_1, None, (0, 255, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    output_image_2 = cv.drawKeypoints(image_2, keypoints_2, None, (0, 255, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

     # convert to rgb for matplotlib
    output_image_1 = cv.cvtColor(output_image_1, cv.COLOR_BGR2RGB)
    output_image_2 = cv.cvtColor(output_image_2, cv.COLOR_BGR2RGB)


    axes[0].imshow(output_image_1)
    axes[1].imshow(output_image_2)
    axes[0].axis("off")
    axes[1].axis("off")
    
    plt.tight_layout(pad=2.0)
    plt.show()

                    # image processing 

def process_image(image, type, target_size: tuple[int, int] = (1280, 960)):

    if type == "GRAYSCALE":
        image = cv.resize(image, target_size, interpolation=cv.INTER_AREA)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return image
    elif type == "CLAHE": # used CLAHE normalization from Soenksen paper SPL_UD_DL

        image = cv.resize(image, target_size, interpolation=cv.INTER_AREA)
        img_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)

        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_hsv[:,:,2] = clahe.apply(img_hsv[:,:,2])

        img = cv.cvtColor(img_hsv, cv.COLOR_HSV2RGB)

        return img
    elif type == "RESIZE":
        return cv.resize(image,target_size,interpolation=cv.INTER_AREA)
    else:
        raise NotImplementedError(f"Unknown process image type {type}")


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
    
    mask = masks[0][0][best_mask_index].numpy().astype(np.uint8) * 255

    return mask

def add_padding(image, padding=300):
   """
   returns a padded image, done before registration
   """
   return cv.copyMakeBorder(
      image,
      padding,
      padding,
      padding,
      padding,
      cv.BORDER_CONSTANT,
      value=0
   )

                        # FEATURE DETECTION

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
      raise NotImplementedError("Detection type not found")

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

                        # FEATURE MATCHING

def match_features(kpsA, descsA, kpsB, descsB, feature_detection_type, matcher_type="BF"):
    '''
    Goes to the correct matcher.

    matcher_type : "BF" or "FLANN"
    feature_detection_type : "SIFT", "ORB", or "AKAZE"

    Returns: ptsA, ptsB, top_matches
    '''
    if matcher_type == "BF":
        return _bfMatcher(kpsA, descsA, kpsB, descsB, feature_detection_type)
    elif matcher_type == "FLANN":
        return _flannMatcher(kpsA, descsA, kpsB, descsB, feature_detection_type)
    else:
        raise NotImplementedError(f"Unknown matcher_type '{matcher_type}'. Choose 'BF' or 'FLANN'.")


def _flannMatcher(kpsA, descsA, kpsB, descsB, feature):
    
    if feature == "SIFT":
        # k-d tree for sift/ floating point descriptors
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
    else:
        # LSH index for binary descriptors
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

    top_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            top_matches.append(m[0])

    ptsA = np.asarray([kpsA[m.queryIdx].pt for m in top_matches], dtype=np.float32).reshape(-1, 1, 2)
    ptsB = np.asarray([kpsB[m.trainIdx].pt for m in top_matches], dtype=np.float32).reshape(-1, 1, 2)
                      
    return ptsA, ptsB, top_matches


def _bfMatcher(kpsA, descsA, kpsB, descsB, feature):

    if feature == "SIFT":
        norm = cv.NORM_L2          # using euclidean norm for SIFT
    else:
        norm = cv.NORM_HAMMING     # for binary descriptors we use hamming distance ex. ORB, AKAZE
 
    bf = cv.BFMatcher(norm)

    matches = bf.knnMatch(descsA, descsB, k=2) 

    # lowes ratio test to filter matches
    top_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            top_matches.append(m[0])

    ptsA = np.asarray([kpsA[m.queryIdx].pt for m in top_matches], dtype=np.float32).reshape(-1, 1, 2)
    ptsB = np.asarray([kpsB[m.trainIdx].pt for m in top_matches], dtype=np.float32).reshape(-1, 1, 2)

    return ptsA, ptsB, top_matches



                            # Efficient Loftr

def match_eloftr(image1, image2, model, processor,device, threshold=.1):
    # convert to PIL
    img1 = Image.fromarray(image1)
    img2 = Image.fromarray(image2)
    images = [img1, img2]

    inputs = processor(images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}


    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
   

    image_sizes = [[(img1.height, img1.width), (img2.height, img2.width)]]
    results = processor.post_process_keypoint_matching(
        outputs, image_sizes, threshold=threshold
    )

    # extract matched points
    result = results[0]
    ptsA = result["keypoints0"].cpu().numpy().astype(np.float32).reshape(-1, 1, 2)
    ptsB = result["keypoints1"].cpu().numpy().astype(np.float32).reshape(-1, 1, 2)

    return ptsA, ptsB



def filter_eloftr_matches(ptsA, ptsB, ransacThreshold=8.0):
    threshold = ransacThreshold
    F, mask = cv.findFundamentalMat(
        ptsA.reshape(-1, 2),
        ptsB.reshape(-1, 2),
        cv.USAC_MAGSAC, # used by demo
        threshold,
        0.999
    )
    if mask is None:
        return ptsA, ptsB
    inliers = mask.ravel().astype(bool)
    print(f"Inliers after fundamental: {inliers.sum()} / {len(mask)}")
    return ptsA[inliers].reshape(-1, 1, 2), ptsB[inliers].reshape(-1, 1, 2)


                    # IMAGE WARPING 



def register_image(image_1, image_2, image_1_pts, image_2_pts, ransacThreshold:float, transformation_type="TPS",regularization=5000):
    """
    returns:
        aligned image
        inliers
    """

    if transformation_type == "Affine":
        return affine_transform(image_1,image_2, image_1_pts,image_2_pts,ransacThreshold)
    
    elif transformation_type == "Homograpy":
        return homography(image_1,image_2, image_1_pts,image_2_pts,ransacThreshold)
    
    elif transformation_type == "TPS":
        return thin_plate_spline(image_1,image_2, image_1_pts,image_2_pts,ransacThreshold,regularization)
    
    else:
        raise NotImplementedError(f"Unknown transformation type {transformation_type}, only Affine, Homography, and Thin Plate Spline Supported")



def affine_transform(image_1,image_2,image_1_pts,image_2_pts, ransacThreshold=3.0):
  """
    ransac reprojection threshold controls how strict finding inliers is
  """
  
  (M, mask) = cv.estimateAffine2D(image_2_pts, image_1_pts, method=cv.RANSAC, ransacReprojThreshold=ransacThreshold)

  if M is None:
    print("Tranformation matrix not found")
    return None
 
  print(f'Inlier count: {np.sum(mask)}')
  print(f"Inlier ratio: {np.sum(mask) / len(mask)}")

  (h, w) = image_1.shape[:2]
  aligned_image = cv.warpAffine(image_2, M, (w, h))
  
  return aligned_image, mask



def homography( image_1,image_2,image_1_pts,image_2_pts, ransacThreshold=3.0):
    """
    ransac reprojection threshold controls how strict finding inliers is
    """
  # mask 
    (H, mask) = cv.findHomography(image_2_pts, image_1_pts,cv.RANSAC, ransacReprojThreshold=ransacThreshold)  

    if H is None:
        print("Transformation matrix not found")
        return None

    print(f'Inlier count: {np.sum(mask)}')
    print(f"Inlier ratio: {np.sum(mask) / len(mask)}")

    (h, w) = image_1.shape[:2]
    aligned_image = cv.warpPerspective(image_2, H, (w, h))
  
    return aligned_image, mask



def thin_plate_spline(image_1,image_2,image_1_pts,image_2_pts, ransacThreshold=3.0, regularization=5000):

    # do ransac to get better matches
    ptsA, ptsB, inliers = refine_tps_pts(image_1_pts,image_2_pts,ransacThreshold)

    # regularization controls how much we want it to fit the src image, lower means the warped image will move more
    tps = cv.createThinPlateSplineShapeTransformer(regularization) 

    # create new matches 
    matches = [cv.DMatch(i, i, 0) for i in range(ptsA.shape[1])]

    tps.estimateTransformation(ptsA,ptsB,matches)

    warped_image = tps.warpImage(image_2)

    # get the dimensions of the src image
    h,w = image_1.shape[:2]

    return warped_image[:h,:w], inliers

def refine_tps_pts(ptsA,ptsB,ransacThreshold):

    M, mask = cv.estimateAffine2D(ptsB,ptsA,method=cv.RANSAC, ransacReprojThreshold=ransacThreshold)

    if M is None:
        print("Transformation matrix not found")
        return None

    # creates a boolean mask where inliers are true and outliers are false
    inlier_mask = mask.ravel().astype(bool)

    # keep only matched points that are inliers
    ptsA_refined = ptsA[inlier_mask].reshape(1,-1,2)
    ptsB_refined = ptsB[inlier_mask].reshape(1,-1,2)

    print(f"Inlier Count: {inlier_mask.sum()}")
    print(f"Inlier ratio: {inlier_mask.sum() / len(mask)}")

    return ptsA_refined,ptsB_refined, mask
