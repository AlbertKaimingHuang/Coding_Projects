"""Cylindrical stitching for the Homework 2 five-frame sequences."""

import cv2
import numpy as np


def _image_center(image):
    height, width = image.shape[:2]
    return np.array([(width - 1) / 2, (height - 1) / 2])


def _unit_rays(points, center, focal):
    xy = (np.asarray(points) - center) / focal
    rays = np.column_stack((xy, np.ones(len(xy))))
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _fit_rotation(source, target):
    left, _, right = np.linalg.svd(target.T @ source)
    left[:, -1] *= np.linalg.det(left @ right)
    return left @ right


def _align_rotation(points1, points2, center1, center2, focal, rng):
    if len(points1) < 3:
        raise ValueError("Camera rotation needs at least three matches")

    source = _unit_rays(points1, center1, focal)
    target = _unit_rays(points2, center2, focal)
    points2 = np.asarray(points2)
    best_score = (-1, -np.inf)
    for _ in range(1000):
        sample = rng.choice(len(source), 3, replace=False)
        rotation = _fit_rotation(source[sample], target[sample])
        rays = source @ rotation.T
        depth = rays[:, 2]
        projected = focal * rays[:, :2] / np.maximum(depth[:, None], 1e-6)
        errors = np.linalg.norm(projected + center2 - points2, axis=1)
        inliers = (depth > 0) & (errors < 3)
        score = (int(inliers.sum()), -errors[inliers].sum())
        if score > best_score:
            best_score = score
            best_inliers = inliers

    if best_score[0] < 3:
        raise ValueError("Too few inliers to estimate camera rotation")
    return _fit_rotation(source[best_inliers], target[best_inliers])


def _frame_boundary(image):
    height, width = image.shape[:2]
    x = np.linspace(0, width - 1, 97)
    y = np.linspace(0, height - 1, 97)
    return np.vstack((
        np.column_stack((x, np.zeros_like(x))),
        np.column_stack((x, np.full_like(x, height - 1))),
        np.column_stack((np.zeros_like(y), y)),
        np.column_stack((np.full_like(y, width - 1), y)),
    ))


def _render_cylinder(images, rotations, focal):
    boundaries = []
    for image, rotation in zip(images, rotations):
        rays = _unit_rays(_frame_boundary(image), _image_center(image), focal)
        x, y, z = (rays @ rotation.T).T
        boundaries.append(np.column_stack((
            focal * np.arctan2(x, z), focal * y / np.hypot(x, z),
        )))

    boundary = np.vstack(boundaries)
    u_min, v_min = np.floor(boundary.min(axis=0)).astype(int)
    u_max, v_max = np.ceil(boundary.max(axis=0)).astype(int)
    u, v = np.meshgrid(
        np.arange(u_min, u_max + 1, dtype=np.float32),
        np.arange(v_min, v_max + 1, dtype=np.float32),
    )
    rays = np.stack((np.sin(u / focal), v / focal, np.cos(u / focal)), axis=-1)
    colors = np.zeros((*u.shape, 3), dtype=np.float32)
    weights = np.zeros(u.shape, dtype=np.float32)

    for image, rotation in zip(images, rotations):
        x, y, depth = (rays @ rotation.astype(np.float32)).transpose(2, 0, 1)
        cx, cy = _image_center(image)
        in_front = depth > 1e-6
        safe_depth = np.maximum(depth, 1e-6)
        map_x = np.where(in_front, focal * x / safe_depth + cx, -1).astype(np.float32)
        map_y = np.where(in_front, focal * y / safe_depth + cy, -1).astype(np.float32)

        height, width = image.shape[:2]
        edge_x = np.minimum(map_x, width - 1 - map_x)
        edge_y = np.minimum(map_y, height - 1 - map_y)
        feather = np.clip((np.minimum(edge_x, edge_y) + 1) / 32, 0, 1)
        warped = cv2.remap(image.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
        coverage = cv2.remap(
            np.ones((height, width), dtype=np.float32), map_x, map_y, cv2.INTER_LINEAR,
        )
        colors += warped * feather[..., None]
        weights += coverage * feather

    panorama = colors / np.maximum(weights[..., None], 1e-6)
    return np.uint8(np.clip(panorama, 0, 255))


def generate_cylindrical_panorama(images, feature_matching, focal):
    """Stitch an ordered sequence around its middle frame using one focal length."""
    if not images or not np.isfinite(focal) or focal <= 0:
        raise ValueError("Expected images and a positive focal length in pixels")

    mid = len(images) // 2
    rotations = [None] * len(images)
    rotations[mid] = np.eye(3)
    rng = np.random.default_rng(0)
    right_links = [(i, i - 1) for i in range(mid + 1, len(images))]
    left_links = [(i, i + 1) for i in range(mid - 1, -1, -1)]
    for current, neighbor in right_links + left_links:
        image1, image2 = images[current], images[neighbor]
        points1, points2 = feature_matching(image1, image2)
        relative = _align_rotation(
            points1, points2, _image_center(image1), _image_center(image2), focal, rng,
        )
        rotations[current] = rotations[neighbor] @ relative

    return _render_cylinder(images, rotations, focal)
