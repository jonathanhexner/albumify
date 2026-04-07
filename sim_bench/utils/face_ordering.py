"""
Deterministic face ordering utilities.

Provides standard conventions for ordering detected faces within an image.
"""

import numpy as np
from typing import List, Any


def sort_faces_reading_order(faces: List[Any]) -> List[Any]:
    """
    Sort faces by reading order (top-to-bottom, left-to-right).

    This provides deterministic, intuitive ordering regardless of detection confidence.

    Algorithm:
    1. Group faces into horizontal rows based on Y overlap
    2. Within each row, sort left-to-right by X coordinate
    3. Sort rows top-to-bottom

    Args:
        faces: List of face objects with .bbox attribute (x1, y1, x2, y2)

    Returns:
        Sorted list of faces in reading order

    Example:
        Face layout:     Reading order:
         [A]  [B]         0:A  1:B
         [C]  [D]         2:C  3:D
    """
    if not faces:
        return []

    # Extract bounding boxes
    bboxes = []
    for face in faces:
        if hasattr(face, 'bbox'):
            bbox = face.bbox
            if isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()
            bboxes.append(bbox)
        else:
            # Fallback if no bbox attribute
            return faces

    # Calculate average face height for row grouping
    heights = [bbox[3] - bbox[1] for bbox in bboxes]
    avg_height = np.mean(heights)
    row_threshold = avg_height * 0.5  # Faces overlap >50% vertically = same row

    # Create sort key for each face
    face_keys = []
    for i, bbox in enumerate(bboxes):
        center_y = (bbox[1] + bbox[3]) / 2
        center_x = (bbox[0] + bbox[2]) / 2

        # Assign row number (quantize Y coordinate)
        row = int(center_y / row_threshold)

        face_keys.append((row, center_x, i))  # (row, x_pos, original_index)

    # Sort by row, then by X within row
    sorted_indices = [idx for _, _, idx in sorted(face_keys)]

    # Return faces in reading order
    return [faces[i] for i in sorted_indices]


def sort_faces_by_area(faces: List[Any], descending: bool = True) -> List[Any]:
    """
    Sort faces by bounding box area.

    Args:
        faces: List of face objects with .bbox attribute
        descending: If True, largest first. If False, smallest first.

    Returns:
        Sorted list of faces by area
    """
    if not faces:
        return []

    def get_area(face):
        if hasattr(face, 'bbox'):
            bbox = face.bbox
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return 0

    return sorted(faces, key=get_area, reverse=descending)


def sort_faces_by_confidence(faces: List[Any], descending: bool = True) -> List[Any]:
    """
    Sort faces by detection confidence score.

    This is the default order returned by InsightFace.

    Args:
        faces: List of face objects with .det_score attribute
        descending: If True, highest confidence first

    Returns:
        Sorted list of faces by confidence
    """
    if not faces:
        return []

    def get_score(face):
        if hasattr(face, 'det_score'):
            return face.det_score
        return 0

    return sorted(faces, key=get_score, reverse=descending)


def get_face_ordering_index(faces: List[Any], ordering: str = 'reading_order') -> List[int]:
    """
    Get the index mapping for a specific face ordering convention.

    Args:
        faces: List of face objects
        ordering: One of 'reading_order', 'area', 'confidence', 'detection_order'

    Returns:
        List of indices showing the mapping from new order to original order

    Example:
        original_faces = [faceA, faceB, faceC]
        indices = get_face_ordering_index(original_faces, 'reading_order')
        # indices = [2, 0, 1]  means reading order is [faceC, faceA, faceB]
    """
    if not faces:
        return []

    if ordering == 'detection_order':
        return list(range(len(faces)))

    # Create indexed faces
    indexed_faces = [(i, face) for i, face in enumerate(faces)]

    # Sort according to convention
    if ordering == 'reading_order':
        sorted_faces = sort_faces_reading_order([f for _, f in indexed_faces])
    elif ordering == 'area':
        sorted_faces = sort_faces_by_area([f for _, f in indexed_faces])
    elif ordering == 'confidence':
        sorted_faces = sort_faces_by_confidence([f for _, f in indexed_faces])
    else:
        raise ValueError(f"Unknown ordering: {ordering}")

    # Find original indices
    result = []
    for sorted_face in sorted_faces:
        for orig_idx, orig_face in indexed_faces:
            if orig_face is sorted_face:
                result.append(orig_idx)
                break

    return result
