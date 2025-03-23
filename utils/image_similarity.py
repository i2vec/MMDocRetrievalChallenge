import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def cal_RANSAC_similarity(new_image_path1, new_image_path2):
    # Re-load the new set of images
    # new_image_path1 = "./test_images/img_2.png"
    # new_image_path2 = "./test_images/img_3.png"

    # Read images
    new_image1 = cv2.imread(new_image_path1)
    new_image2 = cv2.imread(new_image_path2)

    # Ensure same size for SSIM and Histogram comparison
    new_image1_resized = cv2.resize(new_image1, (300, 300))
    new_image2_resized = cv2.resize(new_image2, (300, 300))

    # Convert to grayscale for SSIM
    new_image1_gray = cv2.cvtColor(new_image1_resized, cv2.COLOR_BGR2GRAY)
    new_image2_gray = cv2.cvtColor(new_image2_resized, cv2.COLOR_BGR2GRAY)

    # Compute SSIM
    ssim_score_new, _ = ssim(new_image1_gray, new_image2_gray, full=True)

    # Compute color histogram similarity
    hist1_new = cv2.calcHist([new_image1_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2_new = cv2.calcHist([new_image2_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist1_new, hist1_new)
    cv2.normalize(hist2_new, hist2_new)

    hist_similarity_new = cv2.compareHist(hist1_new, hist2_new, cv2.HISTCMP_CORREL)

    # Feature matching with ORB
    orb = cv2.ORB_create(nfeatures=500)
    kp1_new, des1_new = orb.detectAndCompute(new_image1_gray, None)
    kp2_new, des2_new = orb.detectAndCompute(new_image2_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_new = bf.match(des1_new, des2_new)
    matches_new = sorted(matches_new, key=lambda x: x.distance)

    match_ratio_orb_new = len(matches_new) / max(len(kp1_new), len(kp2_new))

    # Feature matching with SIFT + RANSAC
    sift = cv2.SIFT_create()
    kp1_sift_new, des1_sift_new = sift.detectAndCompute(new_image1_gray, None)
    kp2_sift_new, des2_sift_new = sift.detectAndCompute(new_image2_gray, None)

    # FLANN based matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann_new = cv2.FlannBasedMatcher(index_params, search_params)
    matches_sift_new = flann_new.knnMatch(des1_sift_new, des2_sift_new, k=2)

    # Apply Lowe's ratio test
    good_matches_sift_new = [m for m, n in matches_sift_new if m.distance < 0.75 * n.distance]

    match_ratio_sift_new = len(good_matches_sift_new) / max(len(kp1_sift_new), len(kp2_sift_new))

    # Homography transformation using RANSAC
    if len(good_matches_sift_new) > 10:
        src_pts_new = np.float32([kp1_sift_new[m.queryIdx].pt for m in good_matches_sift_new]).reshape(-1, 1, 2)
        dst_pts_new = np.float32([kp2_sift_new[m.trainIdx].pt for m in good_matches_sift_new]).reshape(-1, 1, 2)

        H_new, mask_new = cv2.findHomography(src_pts_new, dst_pts_new, cv2.RANSAC, 5.0)
        inliers_ratio_new = np.sum(mask_new) / len(mask_new)

        similarity_results_new = {
            "SSIM Score": round(ssim_score_new, 4),
            "Histogram Similarity": round(hist_similarity_new, 4),
            "ORB Match Ratio": round(match_ratio_orb_new, 4),
            "SIFT Match Ratio": round(match_ratio_sift_new, 4),
            "Homography Inliers Ratio": round(inliers_ratio_new, 4)
        }
    else:
        similarity_results_new = {
            "SSIM Score": round(ssim_score_new, 4),
            "Histogram Similarity": round(hist_similarity_new, 4),
            "ORB Match Ratio": round(match_ratio_orb_new, 4),
            "SIFT Match Ratio": round(match_ratio_sift_new, 4),
            "Homography Inliers Ratio": 0.0
        }
    return similarity_results_new
    # print(similarity_results_new)
