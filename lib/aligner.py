import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image

try:
    import face_alignment
    _FA_AVAILABLE = True
except ImportError:
    _FA_AVAILABLE = False


# Reference 5-point landmarks for a 256×256 canonical crop.
# (left-eye, right-eye, nose, left-mouth, right-mouth)
_REF_LANDMARKS_256 = np.array([
    [73.5318,  90.3065],
    [182.4682, 90.3065],
    [128.0,    152.9873],
    [85.5172,  202.5765],
    [170.4828, 202.5765],
], dtype=np.float32)


def _get_5pt_from_68(landmarks_68):
    """Collapse 68-point landmarks to 5-point landmarks."""
    left_eye  = landmarks_68[36:42].mean(axis=0)
    right_eye = landmarks_68[42:48].mean(axis=0)
    nose      = landmarks_68[30]
    left_mouth  = landmarks_68[48]
    right_mouth = landmarks_68[54]
    return np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)


class FaceAligner:
    """Detect and align a face image to a canonical 256×256 crop.

    Uses ``face_alignment`` (pip install face-alignment) for landmark
    detection.  Falls back to a centre-crop if the library is unavailable
    or no face is detected.

    Args:
        device (str): ``'cuda'`` or ``'cpu'``.
        output_size (int): size of the square output crop (default 256).
    """

    def __init__(self, device='cuda', output_size=256):
        self.output_size = output_size
        self.ref = _REF_LANDMARKS_256 * (output_size / 256.0)
        self._fa = None

        if _FA_AVAILABLE:
            try:
                self._fa = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.TWO_D,
                    device=device,
                    face_detector='sfd',
                )
            except Exception:
                self._fa = None

        if self._fa is None:
            print(
                '[FaceAligner] face_alignment not available or failed to initialise. '
                'Falling back to centre-crop alignment.'
            )

    # ── public API ────────────────────────────────────────────────────────────

    def align_face(self, image_file: str,
                   alignment_errors_file: str = None,
                   face_detection_errors_file: str = None) -> np.ndarray:
        """Detect, align, and crop the face in *image_file*.

        Args:
            image_file (str): path to an image file.
            alignment_errors_file (str): optional path to a text file where
                alignment failures are logged.
            face_detection_errors_file (str): optional path to a text file
                where detection failures are logged.

        Returns:
            np.ndarray: RGB image of shape (output_size, output_size, 3),
                        dtype uint8.
        """
        img_rgb = np.array(Image.open(image_file).convert('RGB'))

        if self._fa is not None:
            try:
                preds = self._fa.get_landmarks(img_rgb)
                if preds is not None and len(preds) > 0:
                    lm68 = preds[0]  # (68, 2)
                    lm5  = _get_5pt_from_68(lm68)
                    return self._warp(img_rgb, lm5)
            except Exception as exc:
                if alignment_errors_file:
                    self._log(alignment_errors_file,
                              f'{image_file}: alignment error — {exc}')

        # No landmarks found
        if face_detection_errors_file:
            self._log(face_detection_errors_file,
                      f'{image_file}: no face detected')

        return self._centre_crop(img_rgb)

    # ── private helpers ───────────────────────────────────────────────────────

    def _warp(self, img_rgb: np.ndarray, lm5: np.ndarray) -> np.ndarray:
        """Estimate a similarity transform from *lm5* to the reference and
        warp the image."""
        M, _ = cv2.estimateAffinePartial2D(
            lm5, self.ref,
            method=cv2.LMEDS,
        )
        if M is None:
            return self._centre_crop(img_rgb)
        warped = cv2.warpAffine(
            img_rgb, M,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return warped

    def _centre_crop(self, img_rgb: np.ndarray) -> np.ndarray:
        """Fallback: resize and centre-crop to output_size."""
        h, w = img_rgb.shape[:2]
        s = min(h, w)
        top  = (h - s) // 2
        left = (w - s) // 2
        cropped = img_rgb[top:top + s, left:left + s]
        return cv2.resize(cropped, (self.output_size, self.output_size),
                          interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _log(filepath: str, message: str):
        os.makedirs(osp.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'a') as f:
            f.write(message + '\n')
