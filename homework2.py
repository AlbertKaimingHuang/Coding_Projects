# Homework 2: Image Stitching
# Computer Vision (2026 fall)
#
# Framework for image stitching based on Harris corner detection.
# Allowed libraries: numpy, scipy, cv2, matplotlib, os, etc.
# You may add helper functions in a separate utils.py, but do NOT rename
# the functions in this file or change their input/output signatures.
#
# Conventions used throughout this file:
#   - images are np.ndarray of shape (H, W) (grayscale) or (H, W, 3)
#     (BGR, as returned by cv2.imread)
#   - pixel coordinates are (x, y) tuples; note that numpy indexing of an
#     array a uses a[y, x]

import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, spatial

IMGDIR = 'Problem2Images'
OUTDIR = 'outputs'

# ---------------------------------------------------------------------------
# Tunable hyper-parameters (feel free to experiment with them)
# ---------------------------------------------------------------------------
HARRIS_K = 0.04        # Harris constant, usually in [0.04, 0.06]
HARRIS_WIN = 9         # window size of the smoothing in harris_response
CORNER_THRESH = 0.01   # corner threshold = CORNER_THRESH * max(R)
CORNER_MIN_DIST = 5    # minimum distance (pixels) between two corners
MATCH_RATIO = 0.6      # ratio-test threshold in feature_matching
MIN_MATCH_RATE = 0.03  # abort if matches / min(#corners) falls below this


# ---------------------------------------------------------------------------
# Q1.1 Spatial derivatives (10 pts)
# ---------------------------------------------------------------------------
def gradient_x(img):
    '''
    Input:
        img (np.array (H, W) or (H, W, 3)): input image
    Output:
        grad_x (np.array (H, W)): derivative along the x axis
    Hints:
        - convert the image to grayscale first if it has 3 channels
        - compute with float dtype to avoid overflow/underflow
        - consider smoothing the image (e.g. Gaussian) before
          differentiating to suppress noise
        - use mode='reflect' when filtering at the borders
    '''
    # TODO 
    if img.ndim==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = ndimage.sobel(img.astype(np.float64), axis=1, mode="reflect")
    return output


def gradient_y(img):
    '''
    Input:
        img (np.array (H, W) or (H, W, 3)): input image
    Output:
        grad_y (np.array (H, W)): derivative along the y axis
    '''
    # TODO
    if img.ndim==3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = ndimage.sobel(img.astype(np.float64), axis=0, mode="reflect")
    return output


# ---------------------------------------------------------------------------
# Q1.2 Harris response (15 pts)
# ---------------------------------------------------------------------------
def harris_response(img, k, win_s):
    '''
    Input:
        img   (np.array (H, W) or (H, W, 3)): input image
        k     (float): Harris corner constant
        win_s (int): size of the sliding window
    Output:
        R (np.array (H, W)): Harris response of each pixel,
            R = det(M) - k * tr(M)^2
    Hints:
        - call gradient_x / gradient_y from Q1.1
        - smooth the products Ix*Ix, Ix*Iy, Iy*Iy with a Gaussian window
          of size win_s before computing det(M) and tr(M)
        - avoid python "for" loops; use vectorized numpy/scipy operations
    '''
    # TODO
    Ix = gradient_x(img)
    Iy = gradient_y(img)
    A = ndimage.gaussian_filter(Ix * Ix, sigma=1.5, radius = win_s//2, mode="reflect")
    B = ndimage.gaussian_filter(Ix * Iy, sigma=1.5, radius = win_s//2, mode="reflect")
    C = ndimage.gaussian_filter(Iy * Iy, sigma=1.5, radius = win_s//2, mode="reflect")
    R = A*C-B**2-k*(A+C)**2

    return R


# ---------------------------------------------------------------------------
# Q1.3 Corner selection and non-maximal suppression (5 pts)
# ---------------------------------------------------------------------------
def corner_selection(R, th, min_dist):
    '''
    Input:
        R        (np.array (H, W)): pixel-wise Harris response
        th       (float): threshold of R for selecting corners
        min_dist (int): minimum distance of two nearby corners
    Output:
        pixels (list: N): a list of (x, y) tuples of the selected corners,
            e.g. pixels = [(5, 8), (3, 9)]
    Hints:
        - use ndimage.maximum_filter() for non-maximum suppression:
          keep a pixel only if it is the local maximum within a window of
          size (2 * min_dist + 1)
        - np.nonzero returns (row, col) = (y, x); convert to (x, y)
    '''
    # TODO
    M = ndimage.maximum_filter(R, size=2*min_dist+1)
    output = (R >= th) & (R==M)
    a, b = np.nonzero(output)
    return list(zip(b, a))


# ---------------------------------------------------------------------------
# Q2 Gradient histogram descriptor (20 pts)
# ---------------------------------------------------------------------------
def histogram_of_gradients(img, pix):
    '''
    Input:
        img (np.array (H, W) or (H, W, 3)): input image
        pix (list: N): a list of (x, y) tuples of the corners from Q1
    Output:
        features (np.array (N, L)): L-dimensional feature vectors,
            in the same order as pix
    Requirements (no fixed template, hard-code the parameters):
        1. horizontal and vertical gradients (reuse Q1.1)
        2. gradient magnitude and direction (np.arctan2)
        3. for each corner, an m*m grid of cells with m*m pixels each
        4. per-cell histogram over n directions (e.g. 8), weighted by
           magnitude, then normalized
        5. ROTATION INVARIANCE IS REQUIRED: estimate the dominant
           orientation of the patch, rotate the gradients to align with
           it, and build the descriptor in the canonical frame
    '''
    # TODO
    Ix = gradient_x(img)
    Iy = gradient_y(img)
    m = 4
    half = m*m // 2
    index = np.arange(-half, half)+0.5
    dx, dy = np.meshgrid(index, index)

    features = []
    for x, y in pix:
        px = x + dx
        py = y + dy
        patch_Ix = ndimage.map_coordinates(Ix, [py, px], order=1, mode="reflect")
        patch_Iy = ndimage.map_coordinates(Iy, [py, px], order=1, mode="reflect")
        patch_magnitude = np.sqrt(patch_Ix**2+patch_Iy**2)
        patch_theta = np.arctan2(patch_Iy, patch_Ix)
        
        hist, _ = np.histogram(patch_theta, bins=36, range=(-np.pi, np.pi), weights=patch_magnitude)
        average_index = np.argmax(hist)
        average_theta = -np.pi + (average_index+0.5)*np.pi/18
        newx = x + dx*np.cos(average_theta) - dy*np.sin(average_theta)
        newy = y + dx*np.sin(average_theta) + dy*np.cos(average_theta)
        new_Ix = ndimage.map_coordinates(Ix, [newy, newx], order=1, mode="reflect")
        new_Iy = ndimage.map_coordinates(Iy, [newy, newx], order=1, mode="reflect")
        new_theta =  (np.arctan2(new_Iy, new_Ix)-average_theta)% (2*np.pi)
        new_magnitude = np.sqrt(new_Ix**2+new_Iy**2)

        feature = []
        for i in range(m):
            for j in range(m):
                cell_theta = new_theta[i*m:(i+1)*m, j*m:(j+1)*m]
                cell_magnitude = new_magnitude[i*m:(i+1)*m, j*m:(j+1)*m]
                histqwe, _ = np.histogram(cell_theta, bins=8, range=(0, 2*np.pi), weights=cell_magnitude)
                feature.extend(histqwe)
        feature = np.array(feature, dtype=np.float64)
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature /= norm
        features.append(feature)

    return np.array(features, dtype=np.float64).reshape(len(pix), m*m*8)
        








# ---------------------------------------------------------------------------
# Q3 Local feature matching (5 pts) -- provided, do not modify
# ---------------------------------------------------------------------------
def feature_matching(img_1, img_2):
    '''
    Input:
        img_1, img_2 (np.array (H, W, 3)): image pair
    Output:
        pixels_1, pixels_2 (list: N): matched (x, y) tuples such that
            pixels_1[i] in img_1 corresponds to pixels_2[i] in img_2
    Matches are selected with a ratio test: a match is accepted if
    best_dist / second_best_dist <= MATCH_RATIO.
    '''
    R1 = harris_response(img_1, HARRIS_K, HARRIS_WIN)
    R2 = harris_response(img_2, HARRIS_K, HARRIS_WIN)
    cor1 = corner_selection(R1, CORNER_THRESH * np.max(R1), CORNER_MIN_DIST)
    cor2 = corner_selection(R2, CORNER_THRESH * np.max(R2), CORNER_MIN_DIST)
    fea1 = histogram_of_gradients(img_1, cor1)
    fea2 = histogram_of_gradients(img_2, cor2)

    dis = spatial.distance.cdist(fea1, fea2, metric='euclidean')
    pixels_1, pixels_2 = [], []
    p1, p2 = np.shape(dis)
    if p1 < p2:
        # for each corner in image 1, find its best match in image 2
        for p in range(p1):
            best = np.argmin(dis[p])
            dis_min = dis[p][best]
            dis[p][best] = np.max(dis)  # exclude the best to get 2nd best
            if dis_min / np.min(dis[p]) <= MATCH_RATIO:
                pixels_1.append(cor1[p])
                pixels_2.append(cor2[best])
                dis[:, best] = np.max(dis)  # match each corner at most once
    else:
        # symmetric case: iterate over the corners of image 2
        for p in range(p2):
            best = np.argmin(dis[:, p])
            dis_min = dis[best][p]
            dis[best][p] = np.max(dis)
            if dis_min / np.min(dis[:, p]) <= MATCH_RATIO:
                pixels_2.append(cor2[p])
                pixels_1.append(cor1[best])
                dis[best, :] = np.max(dis)

    rate = len(pixels_1) / min(len(cor1), len(cor2))
    assert rate >= MIN_MATCH_RATE, "Fail to Match!"
    return pixels_1, pixels_2


def test_matching(pair=('1_1.jpg', '1_2.jpg'), save_path='test.jpg'):
    '''
    Provided helper: visualize the matches of an image pair side by side.
    Use pair=('4_1.jpg', '4_2.jpg') to check rotation invariance (Q2).
    '''
    img_1 = cv2.imread(f'{IMGDIR}/{pair[0]}')
    img_2 = cv2.imread(f'{IMGDIR}/{pair[1]}')

    img_gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    pixels_1, pixels_2 = feature_matching(img_1, img_2)

    H_1, W_1 = img_gray_1.shape
    H_2, W_2 = img_gray_2.shape

    img = np.zeros((max(H_1, H_2), W_1 + W_2, 3))
    img[:H_1, :W_1, (2, 1, 0)] = img_1 / 255
    img[:H_2, W_1:, (2, 1, 0)] = img_2 / 255

    plt.figure(figsize=(20, 10), dpi=300)
    plt.imshow(img)

    for (x1, y1), (x2, y2) in zip(pixels_1, pixels_2):
        plt.plot([x1, x2 + W_1], [y1, y2])

    # plt.show()
    plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------------------------------
# Q4.1 Homography estimation (10 pts)
# ---------------------------------------------------------------------------
def compute_homography(pixels_1, pixels_2):
    '''
    Input:
        pixels_1, pixels_2 (list: N', N' >= 4): matched (x, y) tuples
    Output:
        homo_mat (np.array (3, 3)): estimated homography matrix mapping
            points of image 1 onto image 2
    Hints:
        - convert the coordinates to homogeneous form
        - build the 2N' x 9 system A h = 0 from the correspondences and
          solve it with SVD:
              U, S, V = np.linalg.svd(A)
              homo_mat = V[-1].reshape(3, 3)
          (equivalently, use the smallest eigenvector of A^T A)
        - normalize so that homo_mat[2, 2] == 1
    '''
    # TODO
    A = []
    for (x, y), (u, v) in zip(pixels_1, pixels_2):
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A, dtype=np.float64)
    U, S, V = np.linalg.svd(A)
    h = V[-1].reshape(3, 3)
    h /= h[2, 2]
    return h


# ---------------------------------------------------------------------------
# Q4.2 Alignment with RANSAC (15 pts)
# ---------------------------------------------------------------------------
def align_pair(pixels_1, pixels_2):
    '''
    Input:
        pixels_1, pixels_2 (list: N'): matched (x, y) tuples
    Output:
        est_homo (np.array (3, 3)): optimal homography matrix
    Requirements:
        - implement RANSAC: repeatedly sample 4 matches, estimate a
          homography with compute_homography, count the inliers under a
          reprojection-error threshold, and keep the largest inlier set
        - refit the homography on all inliers of the best model
        - for the report you also need the inlier set of the best model;
          you may add a helper (e.g. ransac_with_inliers) in utils.py and
          call it here
    '''
    # TODO
    p1 = np.array(pixels_1, dtype=np.float64)
    p2 = np.array(pixels_2, dtype=np.float64)
    times = 1000
    threshold = 3
    best_count = 0
    best_one = None
    points = np.column_stack((p1, np.ones(len(p1))))

    for _ in range(times):
        choose = np.random.choice(len(p1), 4, replace=False)
        h = compute_homography(p1[choose], p2[choose])
        new_points = (h @ points.T).T
        new_points = new_points[:, :2] / new_points[:, 2:3]
        dx = p2[:, 0] - new_points[:, 0]
        dy = p2[:, 1] - new_points[:, 1]
        errors = np.sqrt(dx**2+dy**2)
        inliers = errors < threshold
        count = np.sum(inliers)
        if  count > best_count:
            best_count = count
            best_one = inliers
    
    if best_count < 4:
        raise ValueError("wrong!")
    
    best_h = compute_homography(p1[best_one], p2[best_one])
    return best_h


# ---------------------------------------------------------------------------
# Q4.3 Stitch and blend (bonus 10 pts) -- provided, do not modify
# ---------------------------------------------------------------------------
def stitch_blend(img_1, img_2, est_homo):
    '''
    Input:
        img_1, img_2 (np.array (H, W, 3)): the images to process
        est_homo (np.array (3, 3)): homography mapping img_1 onto img_2
    Output:
        est_img (np.array (H', W', 3)): the blended image
    Pipeline: project the four corners of img_1 with est_homo, convert
    them back to Cartesian coordinates after normalization, and together
    with the corners of img_2 obtain the bounding box of the new image
    plane. Then inverse-warp both images onto the new plane (cv2.remap
    with linear interpolation) and blend them with alpha blending.
    You may rewrite this with a better blending method for the bonus.
    '''
    h1, w1, d1 = np.shape(img_1)  # d = 3 channels
    h2, w2, d2 = np.shape(img_2)
    p1 = est_homo.dot(np.array([0, 0, 1]))
    p2 = est_homo.dot(np.array([0, h1, 1]))
    p3 = est_homo.dot(np.array([w1, 0, 1]))
    p4 = est_homo.dot(np.array([w1, h1, 1]))
    p1 = np.int16(p1 / p1[2])
    p2 = np.int16(p2 / p2[2])
    p3 = np.int16(p3 / p3[2])
    p4 = np.int16(p4 / p4[2])
    x_min = min(0, p1[0], p2[0], p3[0], p4[0])
    x_max = max(w2, p1[0], p2[0], p3[0], p4[0])
    y_min = min(0, p1[1], p2[1], p3[1], p4[1])
    y_max = max(h2, p1[1], p2[1], p3[1], p4[1])
    x_range = np.arange(x_min, x_max + 1, 1)
    y_range = np.arange(y_min, y_max + 1, 1)
    x, y = np.meshgrid(x_range, y_range)
    x = np.float32(x)
    y = np.float32(y)
    homo_inv = np.linalg.pinv(est_homo)
    trans_x = homo_inv[0, 0] * x + homo_inv[0, 1] * y + homo_inv[0, 2]
    trans_y = homo_inv[1, 0] * x + homo_inv[1, 1] * y + homo_inv[1, 2]
    trans_z = homo_inv[2, 0] * x + homo_inv[2, 1] * y + homo_inv[2, 2]
    trans_x = np.float32(trans_x / trans_z)
    trans_y = np.float32(trans_y / trans_z)
    # note: cv2.remap requires CV_32F maps; float64 maps raise an error
    # remap in float32 so that interpolation keeps sub-pixel precision
    est_img_1 = cv2.remap(np.float32(img_1), trans_x, trans_y,
                          cv2.INTER_LINEAR)
    est_img_2 = cv2.remap(np.float32(img_2), x, y, cv2.INTER_LINEAR)
    alpha1 = cv2.remap(np.ones(np.shape(img_1), dtype=np.float32), trans_x,
                       trans_y, cv2.INTER_LINEAR)
    alpha2 = cv2.remap(np.ones(np.shape(img_2), dtype=np.float32), x, y,
                       cv2.INTER_LINEAR)
    # undo the black-border premultiplication: near a frame's border,
    # INTER_LINEAR scales BOTH the remapped color and the coverage mask by
    # the same fractional coverage t (the default border is black), so
    # blending the colors directly would darken a band along every seam
    # by roughly (t^2+1)/(t+1). Dividing the color by its mask restores
    # the true brightness before the weighted average.
    eps = 1e-6
    est_img_1 = est_img_1 / np.maximum(alpha1, eps)
    est_img_2 = est_img_2 / np.maximum(alpha2, eps)
    alpha = alpha1 + alpha2
    alpha[alpha == 0] = 2
    alpha1 = alpha1 / alpha
    alpha2 = alpha2 / alpha
    est_img = est_img_1 * alpha1 + est_img_2 * alpha2
    return est_img


# ---------------------------------------------------------------------------
# Q4.4 Generate a panorama (5 pts) -- provided, do not modify
# ---------------------------------------------------------------------------
def generate_panorama(ordered_img_seq):
    '''
    Input:
        ordered_img_seq (list): ordered list of images (H, W, 3); image i
            overlaps with images i-1 and i+1
    Output:
        est_pano (np.array (H', W', 3)): the panorama image

    Provided pipeline (the middle image serves as the anchor):
      1. match CONSECUTIVE original images -- matching against the growing
         panorama degrades quickly, because repeatedly warped content
         distorts the descriptors (a warped-once frame may yield orders of
         magnitude fewer matches than the original pair)
      2. compose the pairwise homographies into the anchor frame
      3. compute the bounding box of ALL warped frames, then warp every
         frame exactly once onto the global canvas and blend them with
         normalized alpha weights

    Warping each frame only once avoids two pitfalls of iterative
    stitching: resampling artifacts do not accumulate, and the black
    uncovered holes of a partially built panorama never get blended into
    the result (an all-ones mask cannot tell holes from valid pixels).

    Note: a single reference plane cannot represent a field of view of
    180 degrees or more. The provided scenes span almost 360 degrees, so
    this plane-based pipeline degenerates for frames far from the anchor;
    stitching the full sequence requires cylindrical projection (see the
    assignment PDF, Q4.4 and bonus 3).
    '''
    n = len(ordered_img_seq)
    mid = n // 2  # middle anchor

    # H[i] maps image i into the anchor frame
    H = [None] * n
    H[mid] = np.eye(3)
    for j in range(mid + 1, n):
        pix1, pix2 = feature_matching(ordered_img_seq[j],
                                      ordered_img_seq[j - 1])
        H[j] = H[j - 1].dot(align_pair(pix1, pix2))
    for i in range(mid - 1, -1, -1):
        pix1, pix2 = feature_matching(ordered_img_seq[i],
                                      ordered_img_seq[i + 1])
        H[i] = H[i + 1].dot(align_pair(pix1, pix2))

    # bounding box of all warped frames in the anchor coordinate system
    xs, ys = [], []
    for i, img in enumerate(ordered_img_seq):
        h, w = np.shape(img)[:2]
        for cx, cy in ((0, 0), (0, h), (w, 0), (w, h)):
            p = H[i].dot(np.array([cx, cy, 1.0]))
            p = p / p[2]
            xs.append(p[0])
            ys.append(p[1])
    x_min, x_max = int(np.floor(min(xs))), int(np.ceil(max(xs)))
    y_min, y_max = int(np.floor(min(ys))), int(np.ceil(max(ys)))

    x, y = np.meshgrid(np.arange(x_min, x_max + 1),
                       np.arange(y_min, y_max + 1))
    x = np.float32(x)
    y = np.float32(y)

    acc = np.zeros((y_max - y_min + 1, x_max - x_min + 1, 3),
                   dtype=np.float32)
    wsum = np.zeros((y_max - y_min + 1, x_max - x_min + 1, 1),
                    dtype=np.float32)
    for i, img in enumerate(ordered_img_seq):
        # map canvas coordinates back into image i (inverse warping)
        Minv = np.linalg.pinv(H[i])
        tx = Minv[0, 0] * x + Minv[0, 1] * y + Minv[0, 2]
        ty = Minv[1, 0] * x + Minv[1, 1] * y + Minv[1, 2]
        tz = Minv[2, 0] * x + Minv[2, 1] * y + Minv[2, 2]
        tx = np.float32(tx / tz)
        ty = np.float32(ty / tz)
        warped = cv2.remap(np.float32(img), tx, ty, cv2.INTER_LINEAR)
        cover = cv2.remap(np.ones(np.shape(img)[:2], dtype=np.float32),
                          tx, ty, cv2.INTER_LINEAR)[..., None]
        # undo the black-border premultiplication (see stitch_blend)
        warped = warped / np.maximum(cover, 1e-6)
        acc += warped * cover
        wsum += cover
    est_pano = np.uint8(np.clip(acc / np.maximum(wsum, 1e-6), 0, 255))
    return est_pano


# ---------------------------------------------------------------------------
# Utility: load an ordered image sequence from a folder
# ---------------------------------------------------------------------------
def load_image_sequence(folder):
    '''
    Load all images of a folder in natural order (numbers in file names
    are compared numerically, e.g. 2.jpg comes before 10.jpg).
    '''
    def natural_key(name):
        return [int(t) if t.isdigit() else t
                for t in re.split(r'(\d+)', name)]

    files = sorted(os.listdir(folder), key=natural_key)
    return [cv2.imread(os.path.join(folder, f)) for f in files]


# The provided scenes span almost 360 degrees, which a single reference
# plane cannot represent (see Q4.4 in the assignment PDF). For the base
# points it is sufficient to stitch any 5 consecutive images of a scene;
# the demo below uses the middle 5 frames. A full 360-degree panorama
# needs cylindrical projection (bonus 3).
BASE_MAX_FRAMES = 5

PANORAMA_SCENES = {
    1: 'grail',
    2: 'library',
    3: 'parrington',
    4: 'Xue-Mountain-Entrance',
}


if __name__ == '__main__':
    os.makedirs(OUTDIR, exist_ok=True)

    # sanity check of Q1-Q3 on the default pair
    test_matching()
    # rotation/scale test pair for Q2 (rotation invariance)
    test_matching(pair=('4_1.jpg', '4_2.jpg'), save_path='test_pair4.jpg')

    # stitch the three validation pairs
    for bid in (1, 2, 3):
        for ext in ('.jpg', '.JPG'):
            path_1 = f'{IMGDIR}/{bid}_1{ext}'
            if os.path.exists(path_1):
                break
        path_2 = path_1.replace('_1', '_2')
        img_1 = cv2.imread(path_1)
        img_2 = cv2.imread(path_2)
        pix1, pix2 = feature_matching(img_1, img_2)
        homo = align_pair(pix1, pix2)
        blend = stitch_blend(img_1, img_2, homo)
        cv2.imwrite(f'{OUTDIR}/blend_{bid}.png', np.uint8(blend))

    # stitch the four provided scenes
    for pid, scene in PANORAMA_SCENES.items():
        img_list = load_image_sequence(f'{IMGDIR}/panoramas/{scene}')
        if len(img_list) > BASE_MAX_FRAMES:
            mid = len(img_list) // 2
            half = BASE_MAX_FRAMES // 2
            img_list = img_list[mid - half: mid - half + BASE_MAX_FRAMES]
        try:
            pano = generate_panorama(img_list)
            cv2.imwrite(f'{OUTDIR}/panorama_{pid}.png', pano)
        except Exception as e:
            print(f'scene "{scene}": plane-based stitching failed ({e}). '
                  f'A full 360-degree panorama needs cylindrical '
                  f'projection (see Q4.4 and bonus 3).')
