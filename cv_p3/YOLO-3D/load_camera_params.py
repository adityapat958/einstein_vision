#!/usr/bin/env python3
import os
import json
import numpy as np
from pathlib import Path


def load_camera_params(params_file):
    """
    Load camera parameters from a JSON or MATLAB .mat file.

    For .mat files the function extracts the front-camera intrinsics from the
    calibration struct that OpenCV / MATLAB camera-calibrator toolboxes produce.
    The result is always a plain dict with the same keys as the JSON variant so
    the rest of the codebase stays unchanged.

    Args:
        params_file (str): Path to a .json or .mat camera-parameters file.

    Returns:
        dict | None: Camera parameters or None on failure.
    """
    if not os.path.exists(params_file):
        print(f"Warning: Camera parameters file {params_file} not found. Using default parameters.")
        return None

    ext = Path(params_file).suffix.lower()

    if ext == ".mat":
        return _load_from_mat(params_file)
    else:
        return _load_from_json(params_file)


def _load_from_json(params_file):
    try:
        with open(params_file, 'r') as f:
            params = json.load(f)

        params['camera_matrix'] = np.array(params['camera_matrix'])
        params['dist_coeffs'] = np.array(params['dist_coeffs'])
        params['projection_matrix'] = np.array(params['projection_matrix'])

        print(f"Loaded camera parameters from {params_file}")
        print(f"Camera matrix:\n{params['camera_matrix']}")
        return params

    except Exception as e:
        print(f"Error loading camera parameters from JSON: {e}")
        return None


def _load_from_mat(params_file):
    """
    Load front-camera intrinsics from a MATLAB calibration .mat file.

    The .mat file produced by the MATLAB camera calibrator (or OpenCV's
    cv.FileStorage) typically contains a struct with fields like:
      KK / K / cameraMatrix  — 3×3 intrinsic matrix
      kc / dist_coeffs / distortionCoefficients — distortion coefficients
    This function tries several common field names and falls back to the known
    hardcoded intrinsics if parsing fails.
    """
    try:
        import scipy.io
        mat = scipy.io.loadmat(params_file, squeeze_me=True, struct_as_record=False)

        # --- camera matrix ---
        K = None
        for key in ('KK', 'K', 'cameraMatrix', 'IntrinsicMatrix', 'fc'):
            if key in mat:
                raw = mat[key]
                arr = np.array(raw, dtype=np.float64)
                if arr.shape == (3, 3):
                    K = arr
                    break
                # MATLAB calibrator stores fc=[fx,fy] + cc=[cx,cy] separately
                if key == 'fc' and arr.shape == (2,):
                    cc = np.array(mat.get('cc', [0, 0]), dtype=np.float64)
                    K = np.array([[arr[0], 0, cc[0]],
                                  [0, arr[1], cc[1]],
                                  [0,      0,    1]], dtype=np.float64)
                    break

        if K is None:
            print("[WARN] Could not parse K from .mat — using known front-camera intrinsics.")
            K = np.array([[1594.7, 0, 654.3],
                          [0, 1607.7, 413.4],
                          [0,      0,     1]], dtype=np.float64)

        # --- distortion coefficients ---
        dist = None
        for key in ('kc', 'dist_coeffs', 'distortionCoefficients', 'RadialDistortion'):
            if key in mat:
                dist = np.array(mat[key], dtype=np.float64).ravel()
                break
        if dist is None:
            dist = np.zeros(5, dtype=np.float64)

        # Build projection matrix P = K @ [I | 0]
        P = np.hstack([K, np.zeros((3, 1), dtype=np.float64)])

        params = {
            'camera_matrix': K,
            'dist_coeffs': dist,
            'projection_matrix': P,
            'image_width': int(mat.get('nx', mat.get('image_width', 1280))),
            'image_height': int(mat.get('ny', mat.get('image_height', 960))),
            'reprojection_error': float(mat.get('rpe', mat.get('reprojection_error', 0.0))),
        }

        print(f"Loaded camera parameters from .mat: {params_file}")
        print(f"Camera matrix:\n{K}")
        return params

    except ImportError:
        print("[ERROR] scipy is not installed. Cannot load .mat file. "
              "Install with: pip install scipy")
        return None
    except Exception as e:
        print(f"Error loading camera parameters from .mat: {e}")
        return None


def export_params_to_json(params, output_path):
    """
    Serialise camera params dict to a JSON file for reuse across scripts.

    Args:
        params (dict): Output of load_camera_params().
        output_path (str): Destination .json path.
    """
    serialisable = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in params.items()
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"Camera parameters exported to {output_path}")

def create_projection_matrix(camera_matrix, R=None, t=None):
    """
    Create a projection matrix from camera intrinsic and extrinsic parameters.
    
    Args:
        camera_matrix (numpy.ndarray): Camera intrinsic matrix (3x3)
        R (numpy.ndarray): Rotation matrix (3x3)
        t (numpy.ndarray): Translation vector (3x1)
        
    Returns:
        numpy.ndarray: Projection matrix (3x4)
    """
    if R is None:
        R = np.eye(3)
    
    if t is None:
        t = np.zeros((3, 1))
    
    # Combine rotation and translation
    RT = np.hstack((R, t))
    
    # Create projection matrix
    projection_matrix = camera_matrix @ RT
    
    return projection_matrix

def apply_camera_params_to_estimator(bbox3d_estimator, params):
    """
    Apply camera parameters to a 3D bounding box estimator.
    
    Args:
        bbox3d_estimator: BBox3DEstimator instance
        params (dict): Dictionary containing camera parameters
        
    Returns:
        bbox3d_estimator: Updated BBox3DEstimator instance
    """
    if params is None:
        print("Warning: No camera parameters provided. Using default parameters.")
        return bbox3d_estimator
    
    # Update camera matrix
    if 'camera_matrix' in params:
        bbox3d_estimator.K = params['camera_matrix']
    
    # Update projection matrix
    if 'projection_matrix' in params:
        bbox3d_estimator.P = params['projection_matrix']
    
    print("Applied camera parameters to 3D bounding box estimator")
    
    return bbox3d_estimator

def main():
    """Example usage of the camera parameter functions."""
    # Configuration variables (modify these as needed)
    # ===============================================
    
    # Input file
    params_file = "camera_params.json"  # Path to camera parameters JSON file
    
    # Camera position (for example purposes)
    camera_height = 1.65  # Camera height above ground in meters
    # ===============================================
    
    # Load camera parameters
    params = load_camera_params(params_file)
    
    if params:
        print("\nCamera Parameters:")
        print(f"Image dimensions: {params['image_width']}x{params['image_height']}")
        print(f"Reprojection error: {params['reprojection_error']}")
        
        # Example of creating a projection matrix with different extrinsic parameters
        print(f"\nExample: Creating a projection matrix with camera raised {camera_height}m above ground")
        R = np.eye(3)
        t = np.array([[0], [camera_height], [0]])  # Camera above ground
        
        projection_matrix = create_projection_matrix(params['camera_matrix'], R, t)
        print(f"New projection matrix:\n{projection_matrix}")

if __name__ == "__main__":
    main() 