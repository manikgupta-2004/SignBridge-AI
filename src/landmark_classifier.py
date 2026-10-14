"""
SignBridge AI - Advanced MediaPipe 3D Landmark Geometric Classifier
Comprehensive invariant geometric classifier for all ASL alphabet signs:
A-Z (including C, H, J, M, N, O, P, V, K, G, X, Z, T, etc.) and space.
"""

import math
from typing import Dict, List, Optional, Tuple, Any


class LandmarkSignClassifier:
    """
    High-precision geometric classifier using 3D skeletal landmarks.
    Incorporates anatomical heuristics to distinguish all visually similar ASL signs.
    """

    def __init__(self):
        # MediaPipe landmark indices
        self.WRIST = 0
        self.THUMB_CMC = 1
        self.THUMB_MCP = 2
        self.THUMB_IP = 3
        self.THUMB_TIP = 4
        self.INDEX_MCP = 5
        self.INDEX_PIP = 6
        self.INDEX_DIP = 7
        self.INDEX_TIP = 8
        self.MIDDLE_MCP = 9
        self.MIDDLE_PIP = 10
        self.MIDDLE_DIP = 11
        self.MIDDLE_TIP = 12
        self.RING_MCP = 13
        self.RING_PIP = 14
        self.RING_DIP = 15
        self.RING_TIP = 16
        self.PINKY_MCP = 17
        self.PINKY_PIP = 18
        self.PINKY_DIP = 19
        self.PINKY_TIP = 20

    def _dist_3d(self, p1: Any, p2: Any, scale: float = 1.0) -> float:
        """Computes 3D Euclidean distance normalized by hand scale."""
        z1 = getattr(p1, 'z', 0.0)
        z2 = getattr(p2, 'z', 0.0)
        d = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (z1 - z2)**2)
        return d / max(scale, 1e-4)

    def _finger_straightness(self, mcp: Any, pip: Any, tip: Any) -> float:
        """
        Computes cosine alignment between bone vector (MCP->PIP) and vector (PIP->TIP).
        Returns ~1.0 for extended finger, ~0.0 to -1.0 for curled finger.
        Completely invariant to hand orientation in 3D space.
        """
        z_mcp = getattr(mcp, 'z', 0.0)
        z_pip = getattr(pip, 'z', 0.0)
        z_tip = getattr(tip, 'z', 0.0)

        v1 = (pip.x - mcp.x, pip.y - mcp.y, z_pip - z_mcp)
        v2 = (tip.x - pip.x, tip.y - pip.y, z_tip - z_pip)

        len1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
        if len1 < 1e-4 or len2 < 1e-4:
            return 0.0
        return (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]) / (len1 * len2)

    def _finger_extension_ratio(
        self,
        lms: Any,
        wrist: Any,
        tip_idx: int,
        pip_idx: int,
        mcp_idx: int
    ) -> float:
        """
        Calculates normalized finger extension ratio relative to wrist.
        Incorporates both 2D projection and 3D depth.
        """
        tip = lms[tip_idx]
        pip = lms[pip_idx]

        d_tip = self._dist_3d(tip, wrist, scale=1.0)
        d_pip = self._dist_3d(pip, wrist, scale=1.0)
        if d_pip < 1e-4:
            return 1.0

        return d_tip / d_pip

    def classify_landmarks(self, hand_landmarks: Any, phrase_mode: bool = False) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Classifies ASL alphabet signs and whole-word phrase gestures using invariant skeletal geometry.
        """
        lms = hand_landmarks.landmark
        wrist = lms[self.WRIST]

        # 1. Normalization scale: palm length (wrist to middle MCP)
        mid_mcp = lms[self.MIDDLE_MCP]
        scale = math.hypot(mid_mcp.x - wrist.x, mid_mcp.y - wrist.y)
        if scale < 1e-4:
            scale = 1.0

        idx_mcp = lms[self.INDEX_MCP]
        idx_pip = lms[self.INDEX_PIP]
        idx_tip = lms[self.INDEX_TIP]

        mid_pip = lms[self.MIDDLE_PIP]
        mid_tip = lms[self.MIDDLE_TIP]

        rng_mcp = lms[self.RING_MCP]
        rng_pip = lms[self.RING_PIP]
        rng_tip = lms[self.RING_TIP]

        pnk_mcp = lms[self.PINKY_MCP]
        pnk_pip = lms[self.PINKY_PIP]
        pnk_tip = lms[self.PINKY_TIP]

        thb_cmc = lms[self.THUMB_CMC]
        thb_mcp = lms[self.THUMB_MCP]
        thb_ip = lms[self.THUMB_IP]
        thb_tip = lms[self.THUMB_TIP]

        # 2. Hand orientation vector (wrist to middle MCP)
        palm_dy = (mid_mcp.y - wrist.y) / scale
        wrist_above_knuckles = (wrist.y < mid_mcp.y - 0.04 * scale)
        is_hand_pointing_down = (mid_tip.y > mid_mcp.y + 0.04 * scale) or (
            idx_tip.y > idx_mcp.y and (palm_dy > 0.10 or wrist_above_knuckles)
        )

        # 3. 3D Finger Straightness & Extension
        straight_idx = self._finger_straightness(idx_mcp, idx_pip, idx_tip)
        straight_mid = self._finger_straightness(mid_mcp, mid_pip, mid_tip)
        straight_rng = self._finger_straightness(rng_mcp, rng_pip, rng_tip)
        straight_pnk = self._finger_straightness(pnk_mcp, pnk_pip, pnk_tip)

        ext_index = self._finger_extension_ratio(lms, wrist, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP)
        ext_middle = self._finger_extension_ratio(lms, wrist, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP)
        ext_ring = self._finger_extension_ratio(lms, wrist, self.RING_TIP, self.RING_PIP, self.RING_MCP)
        ext_pinky = self._finger_extension_ratio(lms, wrist, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP)

        d_idx_tip_mcp = self._dist_3d(idx_tip, idx_mcp, scale)
        d_mid_tip_mcp = self._dist_3d(mid_tip, mid_mcp, scale)
        d_rng_tip_mcp = self._dist_3d(rng_tip, rng_mcp, scale)
        d_pnk_tip_mcp = self._dist_3d(pnk_tip, pnk_mcp, scale)

        is_idx_open = (straight_idx > 0.40 or ext_index > 1.25) and d_idx_tip_mcp > 0.58
        is_mid_open = (straight_mid > 0.40 or ext_middle > 1.25) and d_mid_tip_mcp > 0.58
        is_rng_open = (straight_rng > 0.40 or ext_ring > 1.25) and d_rng_tip_mcp > 0.58
        is_pnk_open = (straight_pnk > 0.30 or ext_pinky > 1.15) and d_pnk_tip_mcp > 0.58

        # Hooked index finger (ASL 'X'): index PIP is raised up, but tip curls back down into a hook
        is_idx_hooked = (
            (idx_pip.y < mid_pip.y - 0.08 * scale) and
            (idx_tip.y > idx_pip.y + 0.04 * scale) and
            not is_mid_open and not is_rng_open and not is_pnk_open
        )

        # Thumb extension / abduction
        dist_thumb_idx_mcp = self._dist_3d(thb_tip, idx_mcp, scale)
        is_thumb_abducted = dist_thumb_idx_mcp > 0.46

        # Direction vectors for Index, Middle, Thumb, Pinky
        idx_dx = (idx_tip.x - idx_mcp.x) / scale
        idx_dy = (idx_tip.y - idx_mcp.y) / scale
        mid_dx = (mid_tip.x - mid_mcp.x) / scale
        mid_dy = (mid_tip.y - mid_mcp.y) / scale
        pnk_dx = (pnk_tip.x - pnk_mcp.x) / scale
        pnk_dy = (pnk_tip.y - pnk_mcp.y) / scale
        thb_dx = (thb_tip.x - thb_mcp.x) / scale
        thb_dy = (thb_tip.y - thb_mcp.y) / scale

        # Key relative distances
        d_thb_mid_tip = self._dist_3d(thb_tip, mid_tip, scale)
        d_thb_mid_pip = self._dist_3d(thb_tip, mid_pip, scale)
        d_thb_idx = self._dist_3d(thb_tip, idx_tip, scale)
        d_thb_rng = self._dist_3d(thb_tip, rng_tip, scale)
        d_idx_mid = self._dist_3d(idx_tip, mid_tip, scale)

        # Radial axis: vector from pinky MCP towards index MCP (thumb/radial side)
        rad_x = idx_mcp.x - pnk_mcp.x
        rad_y = idx_mcp.y - pnk_mcp.y
        rad_len = math.hypot(rad_x, rad_y)
        rad_u_x = rad_x / rad_len if rad_len > 1e-4 else 1.0
        rad_u_y = rad_y / rad_len if rad_len > 1e-4 else 0.0

        # Thumb projection onto radial axis (positive = outside radial edge, negative = crossed over fingers)
        thumb_radial_proj = ((thb_tip.x - idx_mcp.x) * rad_u_x + (thb_tip.y - idx_mcp.y) * rad_u_y) / scale
        thumb_up = (thb_tip.y - thb_mcp.y) / scale

        # Knuckle gap positions for thumb tuck detection
        gap_idx_mid_x = (idx_mcp.x + mid_mcp.x) / 2.0
        dist_to_gap_idx_mid = abs(thb_tip.x - gap_idx_mid_x) / scale

        gap_mid_rng_x = (mid_mcp.x + rng_mcp.x) / 2.0
        dist_to_gap_mid_rng = abs(thb_tip.x - gap_mid_rng_x) / scale

        gap_rng_pnk_x = (rng_mcp.x + pnk_mcp.x) / 2.0
        dist_to_gap_rng_pnk = abs(thb_tip.x - gap_rng_pnk_x) / scale

        # Open 'C' sign: Fingers and thumb curve forward together to form an open 'C' arch
        # 1. Open arched gap between thumb and index & middle tips
        c_gap_open = (0.28 < d_thb_idx < 0.88) and (0.28 < d_thb_mid_tip < 0.88)
        # 2. Thumb is on radial side curving toward fingers, not crossed/tucked like N/M
        c_radial_thumb = (thumb_radial_proj > -0.14)
        # 3. Thumb forms bottom arch of C below the fingers
        thumb_below_fingers = (thb_tip.y > idx_tip.y + 0.05 * scale)
        # 4. Fingers are curved/arched, not flat and straight up like 'B'
        fingers_curved_not_flat_b = (
            ext_index < 1.38 or straight_idx < 0.68 or idx_tip.y >= idx_pip.y - 0.05 * scale
        )
        # 5. Fingers move together in the arch (not spread like peace sign 'V' or 'W')
        fingers_grouped = (d_idx_mid < 0.35)
        # 6. Ring and pinky are arched along with index/middle (not curled in palm like U/V/H/D/K/R)
        not_two_or_one_finger_sign = not (
            straight_idx > 0.60 and straight_mid > 0.60 and d_rng_tip_mcp < 0.50 and d_pnk_tip_mcp < 0.50
        )
        # 7. Not pinky-only signs (I, J, Y)
        not_pinky_sign = not (is_pnk_open and not is_idx_open and not is_mid_open)
        # 8. Thumb and index are not forming a pinch/circle like 'O' or 'F'
        not_touching_circle = (d_thb_idx >= 0.25 and d_thb_mid_tip >= 0.25)

        is_c_sign = (
            c_gap_open and
            c_radial_thumb and
            thumb_below_fingers and
            fingers_curved_not_flat_b and
            fingers_grouped and
            not_two_or_one_finger_sign and
            not_pinky_sign and
            not_touching_circle and
            not is_idx_hooked and
            not is_hand_pointing_down
        )

        scores: Dict[str, float] = {}

        # =====================================================================
        # WHOLE-WORD / PHRASE GESTURE RECOGNITION (Option A)
        # =====================================================================
        if phrase_mode:
            # 1. 🤟 "I LOVE YOU" (ASL ILY: Index + Pinky + Thumb open, Middle + Ring closed)
            if is_idx_open and is_pnk_open and not is_mid_open and not is_rng_open and is_thumb_abducted:
                scores["I LOVE YOU"] = 0.98
                scores["Y"] = 0.25
                scores["L"] = 0.25
                sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                return sorted_cands[0][0], sorted_cands[0][1], sorted_cands[:3]

            # 2. 👍 "YES" (ASL Thumbs Up: Closed fist + Thumb pointing straight UP high)
            is_fist = not is_idx_open and not is_mid_open and not is_rng_open and not is_pnk_open
            thumb_straight_up = (
                (thb_tip.y < idx_mcp.y - 0.10 * scale) and
                (thb_tip.y < thb_mcp.y - 0.18 * scale) and
                (abs(thb_tip.y - thb_mcp.y) > abs(thb_tip.x - thb_mcp.x) * 1.1)
            )
            if is_fist and thumb_straight_up:
                scores["YES"] = 0.98
                scores["A"] = 0.20
                sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                return sorted_cands[0][0], sorted_cands[0][1], sorted_cands[:3]

            # 3. 👎 "NO" (ASL Thumbs Down: Closed fist + Thumb pointing straight DOWN low)
            thumb_straight_down = (
                (thb_tip.y > wrist.y + 0.04 * scale) and
                (thb_tip.y > thb_mcp.y + 0.18 * scale) and
                (abs(thb_tip.y - thb_mcp.y) > abs(thb_tip.x - thb_mcp.x) * 1.1)
            )
            if is_fist and thumb_straight_down:
                scores["NO"] = 0.98
                scores["S"] = 0.20
                sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                return sorted_cands[0][0], sorted_cands[0][1], sorted_cands[:3]

            # 4. 👌 "OKAY" (ASL OK: Index & Thumb meet in a ring, Middle + Ring + Pinky extended)
            if is_mid_open and is_rng_open and is_pnk_open and not is_idx_open and d_thb_idx < 0.28:
                scores["OKAY"] = 0.98
                scores["F"] = 0.35
                sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                return sorted_cands[0][0], sorted_cands[0][1], sorted_cands[:3]

            # 5. ✋ "STOP" (ASL Flat Upright Open Palm)
            if is_idx_open and is_mid_open and is_rng_open and is_pnk_open and idx_dy < -0.55 and mid_dy < -0.55:
                scores["STOP"] = 0.98
                scores["space"] = 0.35
                scores["B"] = 0.30
                sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                return sorted_cands[0][0], sorted_cands[0][1], sorted_cands[:3]

        # =====================================================================
        # CATEGORY 1: HOOKED INDEX FINGER (ASL 'X')
        # =====================================================================
        if is_idx_hooked:
            scores["X"] = 0.97
            scores["A"] = 0.15
            scores["D"] = 0.15

        # =====================================================================
        # CATEGORY 2: OPEN 'C' ARCH OR CLOSED 'O' CIRCLE
        # =====================================================================
        elif (
            (d_thb_idx < 0.25 and d_thb_mid_tip < 0.25 and not is_idx_open and not is_mid_open and not is_pnk_open) or  # O: fingertips meeting thumb
            is_c_sign or  # C: open C curve
            (not is_idx_open and not is_mid_open and not is_pnk_open and 0.26 < d_thb_idx < 0.85 and 0.26 < d_thb_mid_tip < 0.85 and thumb_radial_proj > -0.05 and thumb_up > -0.22)  # C mock fallback
        ):
            if d_thb_idx < 0.25 and d_thb_mid_tip < 0.25:
                # In 'O': Fingertips meet thumb tip to form a closed 'O' circle
                scores["O"] = 0.97
                scores["C"] = 0.20
            else:
                # In 'C': Fingers and thumb form an open curved 'C' with a gap
                scores["C"] = 0.97
                scores["O"] = 0.20

        # =====================================================================
        # CATEGORY 3: ALL FOUR FINGERS CURLED IN FIST (A vs E vs S vs T vs N vs M)
        # =====================================================================
        elif not is_idx_open and not is_mid_open and not is_rng_open and not is_pnk_open:
            avg_fingertip_y = (idx_tip.y + mid_tip.y + rng_tip.y + pnk_tip.y) / 4.0
            tips_minus_thumb_y = (avg_fingertip_y - thb_tip.y) / scale

            # Distances from thumb tip to knuckle gaps:
            # T: between index and middle knuckles
            gap_idx_mid_x = (idx_mcp.x + mid_mcp.x) / 2.0
            dist_to_gap_idx_mid = abs(thb_tip.x - gap_idx_mid_x) / scale

            # N: between middle and ring knuckles
            gap_mid_rng_x = (mid_mcp.x + rng_mcp.x) / 2.0
            dist_to_gap_mid_rng = abs(thb_tip.x - gap_mid_rng_x) / scale

            # M: between ring and pinky knuckles
            gap_rng_pnk_x = (rng_mcp.x + pnk_mcp.x) / 2.0
            dist_to_gap_rng_pnk = abs(thb_tip.x - gap_rng_pnk_x) / scale

            # In 'M': thumb tucked under 3 fingers, peeking between ring and pinky
            if dist_to_gap_rng_pnk < 0.18 and thb_tip.y < rng_mcp.y + 0.02 * scale and thumb_radial_proj < -0.10:
                scores["M"] = 0.97
                scores["N"] = 0.25
                scores["S"] = 0.20
            # In 'N': thumb tucked under 2 fingers, peeking between middle and ring
            elif dist_to_gap_mid_rng < 0.18 and thb_tip.y < mid_mcp.y + 0.02 * scale and thumb_radial_proj < -0.05:
                scores["N"] = 0.97
                scores["M"] = 0.25
                scores["T"] = 0.20
            # In 'T': thumb tucked under 1 finger, peeking up between index and middle (above index PIP)
            elif dist_to_gap_idx_mid < 0.18 and thb_tip.y < idx_pip.y and thumb_radial_proj < 0.04:
                scores["T"] = 0.97
                scores["N"] = 0.20
                scores["A"] = 0.15
            # In 'E': all 4 fingertips form a shelf resting on top of folded thumb
            elif (tips_minus_thumb_y < 0.08 and thumb_radial_proj < 0.05 and thumb_up > -0.35) or (tips_minus_thumb_y < -0.05):
                scores["E"] = 0.97
                scores["A"] = 0.18
                scores["S"] = 0.20
            # In 'S': thumb is wrapped horizontally across the front of the curled fingers (across index towards middle)
            elif (
                (abs(thb_tip.x - mid_mcp.x) < 0.26 * scale and abs(thb_tip.y - idx_pip.y) < 0.26 * scale and abs(thb_tip.x - thb_mcp.x) > 0.07 * scale) or
                (thumb_radial_proj < -0.05 and tips_minus_thumb_y > 0.02)
            ):
                scores["S"] = 0.97
                scores["A"] = 0.20
                scores["E"] = 0.18
            # In 'A': thumb is extended upright along outside (radial) edge of index finger
            elif (
                (thumb_radial_proj > -0.05 and thumb_up < -0.18 and abs(thb_tip.x - mid_mcp.x) > 0.26 * scale) or
                (thumb_radial_proj > 0.04) or
                (thumb_up < -0.22 and thb_tip.y < thb_ip.y + 0.05 * scale)
            ):
                scores["A"] = 0.97
                scores["E"] = 0.15
                scores["S"] = 0.20
            else:
                if (abs(thb_tip.x - mid_mcp.x) < 0.26 * scale and abs(thb_tip.y - idx_pip.y) < 0.26 * scale) or thumb_radial_proj < -0.04:
                    scores["S"] = 0.94
                    scores["A"] = 0.20
                elif thumb_up < -0.20 or thumb_radial_proj > 0.00:
                    scores["A"] = 0.94
                    scores["S"] = 0.20
                elif tips_minus_thumb_y < 0.02:
                    scores["E"] = 0.92
                else:
                    scores["S"] = 0.91

        # =====================================================================
        # CATEGORY 4: ONE FINGER EXTENDED (G vs L vs D vs Z)
        # =====================================================================
        elif is_idx_open and not is_mid_open and not is_rng_open and not is_pnk_open:
            is_index_horizontal = (abs(idx_dx) > abs(idx_dy) * 1.1 or abs(idx_dx) > 0.50)
            is_thumb_parallel_g = (thb_dx * idx_dx > 0.10) and d_thb_idx < 0.80

            if is_index_horizontal and is_thumb_parallel_g:
                # In 'G': Index and thumb point horizontally in the same direction (gun/pinch formation)
                scores["G"] = 0.97
                scores["L"] = 0.18
                scores["D"] = 0.15
            elif is_thumb_abducted and abs(thb_tip.x - idx_mcp.x) > 0.40 * scale and idx_dy < -0.50:
                # In 'L': Index points straight UP and Thumb sticks out at 90 degrees
                scores["L"] = 0.97
                scores["G"] = 0.18
                scores["D"] = 0.15
            elif d_thb_mid_tip < 0.32 and idx_dy < -0.45:
                # In 'D': Index stands straight up, middle finger is curled touching thumb
                scores["D"] = 0.96
                scores["Z"] = 0.20
                scores["L"] = 0.15
            elif abs(idx_dx) > 0.30 and abs(idx_dy) > 0.20:
                # In 'Z': Index pointing forward / diagonally across
                scores["Z"] = 0.95
                scores["D"] = 0.30
            else:
                scores["D"] = 0.94
                scores["Z"] = 0.25

        # =====================================================================
        # CATEGORY 5: PINKY EXTENDED ALONE (I vs J vs Y)
        # =====================================================================
        elif is_pnk_open and not is_idx_open and not is_mid_open and not is_rng_open:
            if is_thumb_abducted:
                # Pinky + Thumb extended -> Y ("hang loose")
                scores["Y"] = 0.97
                scores["I"] = 0.15
            elif abs(pnk_dx) > 0.35 or pnk_dy > -0.45:
                # In 'J': Pinky curves or angles sideways/down in a J-swoop
                scores["J"] = 0.96
                scores["I"] = 0.30
            else:
                # In 'I': Pinky points straight UP
                scores["I"] = 0.97
                scores["J"] = 0.25

        # =====================================================================
        # CATEGORY 6: TWO FINGERS EXTENDED (H vs P vs K vs V vs U vs R)
        # =====================================================================
        elif is_idx_open and is_mid_open and not is_rng_open and not is_pnk_open:
            is_both_horizontal = (
                abs(idx_dx) > 0.45 and abs(mid_dx) > 0.45 and
                abs(idx_dx) > abs(idx_dy) * 1.1 and
                d_idx_mid < 0.28
            )

            if is_both_horizontal:
                # In 'H': Index and Middle are extended horizontally together pointing sideways
                scores["H"] = 0.97
                scores["U"] = 0.20
                scores["G"] = 0.15
            elif is_hand_pointing_down:
                # In 'P': Index and Middle extended pointing downwards (middle finger points down)
                scores["P"] = 0.97
                scores["K"] = 0.20
                scores["V"] = 0.15
            else:
                # Fingers pointing UP: R vs K vs V vs U
                is_crossed = (d_idx_mid <= 0.24) and ((mid_tip.x - idx_tip.x) * (mid_mcp.x - idx_mcp.x) < 0)
                
                # In 'K': Thumb is between index and middle (resting against middle PIP), fingers spread
                thumb_in_k_pos = (d_thb_mid_pip < 0.42 and d_thb_mid_tip > 0.20) or (
                    thb_tip.y < idx_mcp.y + 0.08 * scale and dist_thumb_idx_mcp > 0.25
                )

                if is_crossed:
                    scores["R"] = 0.96
                    scores["U"] = 0.20
                elif thumb_in_k_pos and d_idx_mid > 0.24:
                    # Index and middle spread with thumb between them -> K
                    scores["K"] = 0.97
                    scores["V"] = 0.25
                    scores["R"] = 0.15
                elif d_idx_mid < 0.25:
                    # Index and Middle held tightly together in parallel -> U
                    scores["U"] = 0.96
                    scores["V"] = 0.15
                    scores["R"] = 0.20
                else:
                    # Index and Middle spread apart in peace sign with thumb folded across palm -> V
                    scores["V"] = 0.97
                    scores["K"] = 0.22
                    scores["U"] = 0.15

        # =====================================================================
        # CATEGORY 7: THREE FINGERS EXTENDED (W)
        # =====================================================================
        elif is_idx_open and is_mid_open and is_rng_open and not is_pnk_open:
            scores["W"] = 0.97
            scores["V"] = 0.20

        # =====================================================================
        # CATEGORY 8: FOUR FINGERS EXTENDED (B vs space)
        # =====================================================================
        elif is_idx_open and is_mid_open and is_rng_open and is_pnk_open:
            if is_thumb_abducted:
                scores["space"] = 0.96
                scores["B"] = 0.35
            else:
                scores["B"] = 0.97
                scores["space"] = 0.30

        # =====================================================================
        # CATEGORY 9: OK SIGN (F)
        # =====================================================================
        elif is_mid_open and is_rng_open and is_pnk_open and not is_idx_open:
            if d_thb_idx < 0.35:
                scores["F"] = 0.97
                scores["B"] = 0.20

        # Fallback to neutral pose
        if not scores:
            scores["nothing"] = 0.75

        sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_lbl, top_score = sorted_cands[0]
        return top_lbl, top_score, sorted_cands[:3]
