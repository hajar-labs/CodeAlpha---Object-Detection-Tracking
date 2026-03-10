"""
SORT — Simple Online and Realtime Tracking
==========================================
Lightweight re-implementation using NumPy + SciPy only.
Reference: Bewley et al. (2016) https://arxiv.org/abs/1602.00763
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ─── Kalman Filter for a single track ────────────────────────────────────────
class KalmanBoxTracker:
    """
    State vector: [x, y, s, r, dx, dy, ds]
      x, y  = center of bounding box
      s     = scale (area)
      r     = aspect ratio (constant)
      dx,dy,ds = velocities
    """

    count = 0   # global track ID counter

    def __init__(self, bbox):
        """bbox: [x1, y1, x2, y2]"""
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count

        # ── State transition matrix (constant velocity) ──────────────────
        self.F = np.eye(7)
        for i in range(4):
            self.F[i, i + 3] = 1.0      # pos += vel * dt (dt=1)

        # ── Measurement matrix (we observe x,y,s,r) ─────────────────────
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        # ── Covariances ──────────────────────────────────────────────────
        self.R = np.eye(4) * 1.0         # measurement noise
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.          # high uncertainty on velocity
        self.P *= 10.
        self.Q = np.eye(7)               # process noise
        self.Q[4:, 4:] *= 0.01

        # ── State mean & covariance ──────────────────────────────────────
        self.x = np.zeros((7, 1))
        self.x[:4] = self._bbox_to_z(bbox)

        self.hits         = 0
        self.hit_streak   = 0
        self.age          = 0
        self.time_since_update = 0
        self.history      = []

    # ── Coordinate helpers ────────────────────────────────────────────────
    @staticmethod
    def _bbox_to_z(bbox):
        """[x1,y1,x2,y2] → [[cx],[cy],[s],[r]]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        return np.array([[x], [y], [s], [r]])

    @staticmethod
    def _z_to_bbox(z, score=None):
        """[[cx],[cy],[s],[r]] → [x1,y1,x2,y2,(score)]"""
        z = np.array(z).flatten()   # handle both (7,1) state and (4,) input
        w = float(np.sqrt(abs(z[2] * z[3])))
        h = float(abs(z[2])) / w if w > 0 else 1.0
        box = [float(z[0]) - w / 2., float(z[1]) - h / 2.,
               float(z[0]) + w / 2., float(z[1]) + h / 2.]
        if score is None:
            return box
        return box + [score]

    # ── Kalman predict / update ───────────────────────────────────────────
    def predict(self):
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.

        # predict state
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._z_to_bbox(self.x))
        return self.history[-1]

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        z = self._bbox_to_z(bbox)
        y  = z - self.H @ self.x
        S  = self.H @ self.P @ self.H.T + self.R
        K  = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def get_state(self):
        return self._z_to_bbox(self.x)


# ─── IoU helper ──────────────────────────────────────────────────────────────
def iou_batch(bb_test, bb_gt):
    """
    Vectorised IoU between all pairs.
    bb_test: (N,4), bb_gt: (M,4)  →  (N,M) IoU matrix
    """
    bb_gt   = np.expand_dims(bb_gt,   0)   # (1,M,4)
    bb_test = np.expand_dims(bb_test, 1)   # (N,1,4)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * \
                (bb_test[..., 3] - bb_test[..., 1])
    area_gt   = (bb_gt[...,  2] - bb_gt[...,  0]) * \
                (bb_gt[...,  3] - bb_gt[...,  1])

    return inter / (area_test + area_gt - inter + 1e-6)


# ─── Hungarian assignment ─────────────────────────────────────────────────────
def associate_detections(detections, trackers, iou_threshold=0.3):
    """
    Returns:
      matches          : (K,2) array of [det_idx, trk_idx]
      unmatched_dets   : list of unmatched detection indices
      unmatched_trks   : list of unmatched tracker indices
    """
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int),
                list(range(len(detections))), [])

    iou_matrix = iou_batch(detections, trackers)   # (D, T)

    if min(iou_matrix.shape) > 0:
        # Hungarian algorithm (maximise IoU = minimise -IoU)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices  = np.stack([row_ind, col_ind], axis=1)
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_dets = [d for d in range(len(detections))
                      if d not in matched_indices[:, 0]]
    unmatched_trks = [t for t in range(len(trackers))
                      if t not in matched_indices[:, 1]]

    # Filter low-IoU matches
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m)

    matches = np.array(matches, dtype=int) if matches \
              else np.empty((0, 2), dtype=int)

    return matches, unmatched_dets, unmatched_trks


# ─── SORT Tracker ────────────────────────────────────────────────────────────
class SORTTracker:
    """
    SORT multi-object tracker.

    Parameters
    ----------
    max_age       : frames to keep a track alive without updates
    min_hits      : min detections before a track is confirmed
    iou_threshold : IoU threshold for det↔track association
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers      = []
        self.frame_count   = 0
        KalmanBoxTracker.count = 0   # reset global ID counter

    def update(self, dets: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        dets : np.ndarray  shape (N, 5) — [x1, y1, x2, y2, conf]
               Use np.empty((0,5)) when there are no detections.

        Returns
        -------
        np.ndarray  shape (M, 5) — [x1, y1, x2, y2, track_id]
                    Only confirmed tracks are returned.
        """
        self.frame_count += 1

        # ── 1. Predict from all current trackers ─────────────────────────
        trks  = np.zeros((len(self.trackers), 4))
        dead  = []
        for t, trk in enumerate(self.trackers):
            pred = np.array(trk.predict()).flatten()
            trks[t] = pred[:4]
            if np.any(np.isnan(pred)):
                dead.append(t)
        for d in reversed(dead):
            self.trackers.pop(d)
            trks = np.delete(trks, d, axis=0)

        # ── 2. Associate detections with predictions ──────────────────────
        det_boxes = dets[:, :4] if len(dets) > 0 \
                    else np.empty((0, 4))
        matched, unmatched_dets, unmatched_trks = associate_detections(
            det_boxes, trks, self.iou_threshold
        )

        # ── 3. Update matched trackers ────────────────────────────────────
        for d, t in matched:
            self.trackers[t].update(dets[d, :4])

        # ── 4. Create new trackers for unmatched detections ───────────────
        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[d, :4]))

        # ── 5. Collect confirmed tracks & remove dead ones ─────────────────
        results = []
        alive   = []
        for trk in self.trackers:
            state = np.array(trk.get_state()).flatten()   # ensure flat [x1,y1,x2,y2]
            if (trk.time_since_update <= 1) and \
               (trk.hit_streak >= self.min_hits or
                self.frame_count <= self.min_hits):
                results.append([state[0], state[1], state[2], state[3], float(trk.id)])
            if trk.time_since_update <= self.max_age:
                alive.append(trk)

        self.trackers = alive

        return np.array(results, dtype=np.float32) if results \
               else np.empty((0, 5), dtype=np.float32)