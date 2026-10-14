"""
Unit tests for SignBridge AI MediaPipe 3D Landmark Geometric Classifier.
Validates discrimination between visually similar ASL signs:
A, B, C, D, E, G, H, I, J, K, M, N, O, P, T, U, V, X.
"""

import pytest
from src.landmark_classifier import LandmarkSignClassifier


class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class MockLandmarkList:
    def __init__(self, lms):
        self.landmark = lms


def test_classify_u_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)   # wrist
    lms[5] = MockLandmark(0.46, 0.50) # idx MCP
    lms[6] = MockLandmark(0.47, 0.38) # index PIP
    lms[8] = MockLandmark(0.48, 0.20) # index tip
    lms[9] = MockLandmark(0.5, 0.5)   # middle MCP
    lms[10] = MockLandmark(0.51, 0.38) # mid PIP
    lms[12] = MockLandmark(0.52, 0.20) # mid tip
    lms[13] = MockLandmark(0.55, 0.52)
    lms[14] = MockLandmark(0.54, 0.55)
    lms[16] = MockLandmark(0.55, 0.62)
    lms[17] = MockLandmark(0.60, 0.55)
    lms[18] = MockLandmark(0.58, 0.58)
    lms[20] = MockLandmark(0.59, 0.65)
    lms[4] = MockLandmark(0.42, 0.50)

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "U"
    assert conf >= 0.85


def test_classify_v_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)
    lms[5] = MockLandmark(0.44, 0.50)
    lms[6] = MockLandmark(0.44, 0.38)
    lms[8] = MockLandmark(0.36, 0.20) # index spread far
    lms[9] = MockLandmark(0.5, 0.5)
    lms[10] = MockLandmark(0.56, 0.38)
    lms[12] = MockLandmark(0.64, 0.20) # mid spread far
    lms[13] = MockLandmark(0.55, 0.52)
    lms[14] = MockLandmark(0.54, 0.55)
    lms[16] = MockLandmark(0.55, 0.62)
    lms[17] = MockLandmark(0.60, 0.55)
    lms[18] = MockLandmark(0.58, 0.58)
    lms[20] = MockLandmark(0.59, 0.65)
    # Thumb folded across palm
    lms[1] = MockLandmark(0.46, 0.70)
    lms[2] = MockLandmark(0.44, 0.62)
    lms[3] = MockLandmark(0.48, 0.56)
    lms[4] = MockLandmark(0.52, 0.54)

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "V"
    assert conf >= 0.85


def test_classify_a_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)
    lms[9] = MockLandmark(0.5, 0.5)
    lms[5] = MockLandmark(0.42, 0.52)
    lms[6] = MockLandmark(0.42, 0.46)
    lms[8] = MockLandmark(0.42, 0.56) # curled tip
    lms[10] = MockLandmark(0.50, 0.46)
    lms[12] = MockLandmark(0.50, 0.56) # curled
    lms[13] = MockLandmark(0.58, 0.52)
    lms[14] = MockLandmark(0.58, 0.46)
    lms[16] = MockLandmark(0.58, 0.56) # curled
    lms[17] = MockLandmark(0.64, 0.55)
    lms[18] = MockLandmark(0.64, 0.50)
    lms[20] = MockLandmark(0.64, 0.58) # curled
    # Thumb upright on outside edge alongside index finger
    lms[1] = MockLandmark(0.46, 0.72)
    lms[2] = MockLandmark(0.38, 0.64)
    lms[3] = MockLandmark(0.36, 0.54)
    lms[4] = MockLandmark(0.35, 0.44) # thumb tip pointing up

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "A"
    assert conf >= 0.90


def test_classify_e_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)
    lms[9] = MockLandmark(0.5, 0.5)
    lms[5] = MockLandmark(0.42, 0.52)
    lms[6] = MockLandmark(0.42, 0.44)
    lms[8] = MockLandmark(0.42, 0.52) # shelf tip
    lms[10] = MockLandmark(0.50, 0.44)
    lms[12] = MockLandmark(0.50, 0.52)
    lms[13] = MockLandmark(0.58, 0.52)
    lms[14] = MockLandmark(0.58, 0.44)
    lms[16] = MockLandmark(0.58, 0.52)
    lms[17] = MockLandmark(0.64, 0.55)
    lms[18] = MockLandmark(0.64, 0.46)
    lms[20] = MockLandmark(0.64, 0.54)
    # Thumb folded horizontally across palm underneath fingertips
    lms[1] = MockLandmark(0.46, 0.72)
    lms[2] = MockLandmark(0.42, 0.64)
    lms[3] = MockLandmark(0.48, 0.60)
    lms[4] = MockLandmark(0.52, 0.58) # thumb tip below fingertips

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "E"
    assert conf >= 0.90


def test_classify_t_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)
    lms[9] = MockLandmark(0.5, 0.5) # mid MCP
    lms[5] = MockLandmark(0.44, 0.52) # idx MCP
    lms[6] = MockLandmark(0.44, 0.46)
    lms[8] = MockLandmark(0.44, 0.56) # curled
    lms[10] = MockLandmark(0.50, 0.46)
    lms[12] = MockLandmark(0.50, 0.56) # curled
    lms[13] = MockLandmark(0.58, 0.52)
    lms[14] = MockLandmark(0.58, 0.46)
    lms[16] = MockLandmark(0.58, 0.56)
    lms[17] = MockLandmark(0.64, 0.55)
    lms[18] = MockLandmark(0.64, 0.46)
    lms[20] = MockLandmark(0.64, 0.56)
    # Thumb tucked between index and middle knuckles
    lms[1] = MockLandmark(0.46, 0.72)
    lms[2] = MockLandmark(0.43, 0.62)
    lms[3] = MockLandmark(0.46, 0.52)
    lms[4] = MockLandmark(0.47, 0.44) # thumb tip poking between index and middle

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "T"
    assert conf >= 0.90


def test_classify_n_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)
    lms[9] = MockLandmark(0.50, 0.50) # mid MCP
    lms[5] = MockLandmark(0.42, 0.52) # idx MCP
    lms[13] = MockLandmark(0.58, 0.50) # rng MCP
    lms[17] = MockLandmark(0.66, 0.55) # pnk MCP
    # Curled fingers
    for pip, tip in [(6, 8), (10, 12), (14, 16), (18, 20)]:
        lms[pip] = MockLandmark(0.5, 0.46)
        lms[tip] = MockLandmark(0.5, 0.56)
    # Thumb tucked between middle and ring
    lms[1] = MockLandmark(0.46, 0.72)
    lms[2] = MockLandmark(0.46, 0.62)
    lms[3] = MockLandmark(0.50, 0.54)
    lms[4] = MockLandmark(0.54, 0.46) # thumb tip between mid (0.50) and rng (0.58)

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "N"
    assert conf >= 0.90


def test_classify_m_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)
    lms[9] = MockLandmark(0.50, 0.50) # mid MCP
    lms[5] = MockLandmark(0.42, 0.52) # idx MCP
    lms[13] = MockLandmark(0.58, 0.50) # rng MCP
    lms[17] = MockLandmark(0.66, 0.55) # pnk MCP
    # Curled fingers
    for pip, tip in [(6, 8), (10, 12), (14, 16), (18, 20)]:
        lms[pip] = MockLandmark(0.5, 0.46)
        lms[tip] = MockLandmark(0.5, 0.56)
    # Thumb tucked between ring and pinky
    lms[1] = MockLandmark(0.46, 0.72)
    lms[2] = MockLandmark(0.50, 0.62)
    lms[3] = MockLandmark(0.56, 0.54)
    lms[4] = MockLandmark(0.62, 0.48) # thumb tip between rng (0.58) and pnk (0.66)

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "M"
    assert conf >= 0.90


def test_classify_c_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)
    lms[9] = MockLandmark(0.50, 0.50)
    lms[5] = MockLandmark(0.44, 0.50)
    # Fingers forming top arch of C
    lms[6] = MockLandmark(0.46, 0.40)
    lms[8] = MockLandmark(0.50, 0.44) # index tip
    lms[10] = MockLandmark(0.52, 0.40)
    lms[12] = MockLandmark(0.54, 0.45) # middle tip
    lms[14] = MockLandmark(0.56, 0.42)
    lms[16] = MockLandmark(0.58, 0.47)
    lms[17] = MockLandmark(0.64, 0.55)
    lms[18] = MockLandmark(0.60, 0.46)
    lms[20] = MockLandmark(0.62, 0.50)
    # Thumb forming bottom arch of C on radial side (open gap with index)
    lms[1] = MockLandmark(0.44, 0.72)
    lms[2] = MockLandmark(0.38, 0.64)
    lms[3] = MockLandmark(0.36, 0.60)
    lms[4] = MockLandmark(0.36, 0.58) # radial side, open gap with tip

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "C"
    assert conf >= 0.90


def test_classify_o_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)
    lms[9] = MockLandmark(0.50, 0.50)
    lms[5] = MockLandmark(0.44, 0.50)
    # Fingers curving to touch thumb tip
    lms[6] = MockLandmark(0.45, 0.42)
    lms[8] = MockLandmark(0.46, 0.48) # index tip meeting thumb
    lms[10] = MockLandmark(0.49, 0.42)
    lms[12] = MockLandmark(0.47, 0.48) # mid tip meeting thumb
    lms[14] = MockLandmark(0.53, 0.44)
    lms[16] = MockLandmark(0.48, 0.49)
    lms[18] = MockLandmark(0.57, 0.46)
    lms[20] = MockLandmark(0.49, 0.50)
    # Thumb meeting fingertips
    lms[1] = MockLandmark(0.44, 0.70)
    lms[2] = MockLandmark(0.42, 0.60)
    lms[3] = MockLandmark(0.44, 0.52)
    lms[4] = MockLandmark(0.46, 0.48) # thumb tip touching index/middle tips

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "O"
    assert conf >= 0.90


def test_classify_h_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.40, 0.70)
    lms[9] = MockLandmark(0.48, 0.58)
    # Index pointing horizontally right
    lms[5] = MockLandmark(0.45, 0.54)
    lms[6] = MockLandmark(0.54, 0.53)
    lms[8] = MockLandmark(0.65, 0.52)
    # Middle pointing horizontally right parallel to index
    lms[10] = MockLandmark(0.54, 0.56)
    lms[12] = MockLandmark(0.65, 0.55)
    # Ring and pinky curled
    lms[13] = MockLandmark(0.46, 0.62); lms[16] = MockLandmark(0.44, 0.66)
    lms[17] = MockLandmark(0.48, 0.64); lms[20] = MockLandmark(0.46, 0.68)
    lms[1] = MockLandmark(0.42, 0.68)
    lms[2] = MockLandmark(0.44, 0.64)
    lms[3] = MockLandmark(0.46, 0.60)
    lms[4] = MockLandmark(0.48, 0.60)

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "H"
    assert conf >= 0.90


def test_classify_j_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)
    lms[9] = MockLandmark(0.5, 0.5)
    lms[5] = MockLandmark(0.44, 0.50) # idx MCP
    # Other fingers curled
    for pip, tip in [(6, 8), (10, 12), (14, 16)]:
        lms[pip] = MockLandmark(0.5, 0.55)
        lms[tip] = MockLandmark(0.5, 0.62)
    # Pinky curved sideways/hooked in J-trace
    lms[17] = MockLandmark(0.62, 0.54)
    lms[18] = MockLandmark(0.64, 0.38)
    lms[20] = MockLandmark(0.74, 0.36) # pinky tip angled sideways
    lms[4] = MockLandmark(0.44, 0.56) # thumb folded over index

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "J"
    assert conf >= 0.90


def test_classify_x_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)
    lms[9] = MockLandmark(0.5, 0.5)
    # Index finger hooked: PIP raised high, TIP bent down
    lms[5] = MockLandmark(0.44, 0.52)
    lms[6] = MockLandmark(0.43, 0.36) # PIP high above palm
    lms[8] = MockLandmark(0.44, 0.46) # Tip bent down into hook
    # Middle, ring, pinky curled
    lms[10] = MockLandmark(0.50, 0.56)
    lms[12] = MockLandmark(0.50, 0.62)
    lms[13] = MockLandmark(0.58, 0.52)
    lms[14] = MockLandmark(0.58, 0.58)
    lms[16] = MockLandmark(0.58, 0.64)
    lms[17] = MockLandmark(0.64, 0.55)
    lms[18] = MockLandmark(0.64, 0.60)
    lms[20] = MockLandmark(0.64, 0.66)
    lms[1] = MockLandmark(0.46, 0.70)
    lms[2] = MockLandmark(0.42, 0.62)
    lms[3] = MockLandmark(0.44, 0.56)
    lms[4] = MockLandmark(0.46, 0.56)

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "X"
    assert conf >= 0.90


def test_classify_g_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.40, 0.70)
    lms[9] = MockLandmark(0.48, 0.58) # middle MCP
    # Index pointing horizontally right
    lms[5] = MockLandmark(0.45, 0.55)
    lms[6] = MockLandmark(0.52, 0.54)
    lms[8] = MockLandmark(0.62, 0.53) # index points horizontally
    # Thumb pointing horizontally parallel to index
    lms[1] = MockLandmark(0.42, 0.68)
    lms[2] = MockLandmark(0.44, 0.64)
    lms[3] = MockLandmark(0.50, 0.62)
    lms[4] = MockLandmark(0.58, 0.60) # thumb points horizontally parallel
    # Middle, ring, pinky curled
    lms[10] = MockLandmark(0.48, 0.64)
    lms[12] = MockLandmark(0.46, 0.66)
    lms[13] = MockLandmark(0.52, 0.62)
    lms[14] = MockLandmark(0.50, 0.66)
    lms[16] = MockLandmark(0.48, 0.68)
    lms[17] = MockLandmark(0.56, 0.65)
    lms[18] = MockLandmark(0.54, 0.68)
    lms[20] = MockLandmark(0.52, 0.70)

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "G"
    assert conf >= 0.90


def test_classify_d_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)
    lms[9] = MockLandmark(0.5, 0.5)
    # Index finger straight up
    lms[5] = MockLandmark(0.44, 0.50)
    lms[6] = MockLandmark(0.43, 0.35)
    lms[8] = MockLandmark(0.42, 0.18)
    # Middle finger curled touching thumb
    lms[10] = MockLandmark(0.50, 0.58)
    lms[12] = MockLandmark(0.47, 0.58)
    lms[13] = MockLandmark(0.58, 0.52)
    lms[14] = MockLandmark(0.58, 0.60)
    lms[16] = MockLandmark(0.58, 0.66)
    lms[17] = MockLandmark(0.65, 0.56)
    lms[18] = MockLandmark(0.65, 0.62)
    lms[20] = MockLandmark(0.65, 0.68)
    # Thumb touches middle tip
    lms[1] = MockLandmark(0.46, 0.70)
    lms[2] = MockLandmark(0.42, 0.62)
    lms[3] = MockLandmark(0.44, 0.58)
    lms[4] = MockLandmark(0.46, 0.58)

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "D"
    assert conf >= 0.90


def test_classify_k_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)
    lms[9] = MockLandmark(0.5, 0.5)
    # Index finger extended UP
    lms[5] = MockLandmark(0.44, 0.50)
    lms[6] = MockLandmark(0.43, 0.35)
    lms[8] = MockLandmark(0.42, 0.18)
    # Middle finger extended UP/forward
    lms[10] = MockLandmark(0.52, 0.36)
    lms[12] = MockLandmark(0.54, 0.22)
    lms[13] = MockLandmark(0.58, 0.52)
    lms[14] = MockLandmark(0.58, 0.60)
    lms[16] = MockLandmark(0.58, 0.66)
    lms[17] = MockLandmark(0.65, 0.56)
    lms[18] = MockLandmark(0.65, 0.62)
    lms[20] = MockLandmark(0.65, 0.68)
    # Thumb extended between index and middle
    lms[1] = MockLandmark(0.46, 0.70)
    lms[2] = MockLandmark(0.43, 0.60)
    lms[3] = MockLandmark(0.46, 0.48)
    lms[4] = MockLandmark(0.48, 0.38)

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "K"
    assert conf >= 0.90


def test_classify_p_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.3)] * 21
    lms[0] = MockLandmark(0.5, 0.3) # wrist at top
    lms[9] = MockLandmark(0.5, 0.5) # middle MCP below wrist
    # Index pointing forward/down
    lms[5] = MockLandmark(0.46, 0.50)
    lms[6] = MockLandmark(0.45, 0.62)
    lms[8] = MockLandmark(0.44, 0.78)
    # Middle pointing DOWN
    lms[10] = MockLandmark(0.52, 0.64)
    lms[12] = MockLandmark(0.54, 0.78)
    # Ring and pinky curled into palm
    lms[13] = MockLandmark(0.58, 0.50)
    lms[14] = MockLandmark(0.58, 0.56)
    lms[16] = MockLandmark(0.58, 0.48)
    lms[17] = MockLandmark(0.65, 0.48)
    lms[18] = MockLandmark(0.65, 0.54)
    lms[20] = MockLandmark(0.65, 0.46)
    # Thumb positioned between them
    lms[1] = MockLandmark(0.46, 0.40)
    lms[2] = MockLandmark(0.43, 0.50)
    lms[3] = MockLandmark(0.46, 0.62)
    lms[4] = MockLandmark(0.48, 0.70)

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "P"
    assert conf >= 0.90


def test_classify_s_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)
    lms[9] = MockLandmark(0.5, 0.5)
    # Curled fist fingers
    lms[5] = MockLandmark(0.42, 0.52)
    lms[6] = MockLandmark(0.42, 0.46)
    lms[8] = MockLandmark(0.42, 0.58) # curled tip
    lms[10] = MockLandmark(0.50, 0.46)
    lms[12] = MockLandmark(0.50, 0.58) # curled
    lms[13] = MockLandmark(0.58, 0.52)
    lms[14] = MockLandmark(0.58, 0.46)
    lms[16] = MockLandmark(0.58, 0.58) # curled
    lms[17] = MockLandmark(0.64, 0.55)
    lms[18] = MockLandmark(0.64, 0.50)
    lms[20] = MockLandmark(0.64, 0.58) # curled
    # Thumb wrapped horizontally across front of index and middle fingers
    lms[1] = MockLandmark(0.46, 0.72)
    lms[2] = MockLandmark(0.38, 0.64)
    lms[3] = MockLandmark(0.44, 0.54)
    lms[4] = MockLandmark(0.50, 0.52) # thumb tip across middle finger at PIP height

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "S"
    assert conf >= 0.90


def test_classify_c_curved_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)
    lms[9] = MockLandmark(0.5, 0.5)
    # Fingers curved forward in open C arch (d_idx_tip_mcp > 0.58)
    lms[5] = MockLandmark(0.44, 0.50)
    lms[6] = MockLandmark(0.44, 0.38) # PIP forward/up
    lms[8] = MockLandmark(0.48, 0.36) # Tip forward in arch
    lms[10] = MockLandmark(0.50, 0.38)
    lms[12] = MockLandmark(0.52, 0.37)
    lms[13] = MockLandmark(0.56, 0.50)
    lms[14] = MockLandmark(0.56, 0.40)
    lms[16] = MockLandmark(0.56, 0.40)
    lms[17] = MockLandmark(0.62, 0.52)
    lms[18] = MockLandmark(0.62, 0.42)
    lms[20] = MockLandmark(0.62, 0.43)
    # Thumb forming bottom arch of C with open gap
    lms[1] = MockLandmark(0.44, 0.70)
    lms[2] = MockLandmark(0.38, 0.62)
    lms[3] = MockLandmark(0.40, 0.55)
    lms[4] = MockLandmark(0.44, 0.52) # thumb tip opposing finger tips with ~0.45 gap

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "C"
    assert conf >= 0.90


def test_classify_s_sign_tight_wrap():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)
    lms[9] = MockLandmark(0.5, 0.5)
    # Curled fist fingers
    lms[5] = MockLandmark(0.42, 0.52)
    lms[6] = MockLandmark(0.42, 0.46)
    lms[8] = MockLandmark(0.42, 0.58)
    lms[10] = MockLandmark(0.50, 0.46)
    lms[12] = MockLandmark(0.50, 0.58)
    lms[13] = MockLandmark(0.58, 0.52)
    lms[14] = MockLandmark(0.58, 0.46)
    lms[16] = MockLandmark(0.58, 0.58)
    lms[17] = MockLandmark(0.64, 0.55)
    lms[18] = MockLandmark(0.64, 0.50)
    lms[20] = MockLandmark(0.64, 0.58)
    # Thumb wrapping across index finger towards middle
    lms[1] = MockLandmark(0.46, 0.72)
    lms[2] = MockLandmark(0.38, 0.64)
    lms[3] = MockLandmark(0.42, 0.54)
    lms[4] = MockLandmark(0.47, 0.50) # thumb resting right in front of index/mid

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "S"
    assert conf >= 0.90


def test_classify_c_profile_view():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)
    lms[9] = MockLandmark(0.5, 0.5)
    # Fingers arching in profile C
    lms[5] = MockLandmark(0.50, 0.48)
    lms[6] = MockLandmark(0.42, 0.42)
    lms[8] = MockLandmark(0.38, 0.44)
    lms[10] = MockLandmark(0.43, 0.43)
    lms[12] = MockLandmark(0.39, 0.45)
    lms[13] = MockLandmark(0.52, 0.52)
    lms[14] = MockLandmark(0.45, 0.44)
    lms[16] = MockLandmark(0.40, 0.46)
    lms[17] = MockLandmark(0.54, 0.54)
    lms[18] = MockLandmark(0.47, 0.46)
    lms[20] = MockLandmark(0.42, 0.48)
    # Thumb arching at bottom of C with open gap
    lms[1] = MockLandmark(0.50, 0.72)
    lms[2] = MockLandmark(0.50, 0.64)
    lms[3] = MockLandmark(0.44, 0.62)
    lms[4] = MockLandmark(0.40, 0.58)

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms))
    assert pred == "C"
    assert conf >= 0.90


def test_classify_ily_phrase_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)   # wrist
    lms[9] = MockLandmark(0.5, 0.5)   # mid MCP
    # Index open & straight
    lms[5] = MockLandmark(0.44, 0.50)
    lms[6] = MockLandmark(0.44, 0.38)
    lms[8] = MockLandmark(0.44, 0.20)
    # Middle closed
    lms[10] = MockLandmark(0.50, 0.52)
    lms[12] = MockLandmark(0.50, 0.60)
    # Ring closed
    lms[13] = MockLandmark(0.55, 0.52)
    lms[14] = MockLandmark(0.55, 0.54)
    lms[16] = MockLandmark(0.55, 0.62)
    # Pinky open & straight
    lms[17] = MockLandmark(0.60, 0.52)
    lms[18] = MockLandmark(0.60, 0.40)
    lms[20] = MockLandmark(0.60, 0.22)
    # Thumb abducted outward
    lms[1] = MockLandmark(0.48, 0.72)
    lms[2] = MockLandmark(0.42, 0.66)
    lms[3] = MockLandmark(0.35, 0.60)
    lms[4] = MockLandmark(0.26, 0.55)

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms), phrase_mode=True)
    assert pred == "I LOVE YOU"
    assert conf >= 0.95


def test_classify_thumbs_up_yes_phrase_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)   # wrist
    lms[9] = MockLandmark(0.5, 0.5)   # mid MCP
    # All 4 fingers curled into fist
    lms[5] = MockLandmark(0.46, 0.50)
    lms[6] = MockLandmark(0.46, 0.54)
    lms[8] = MockLandmark(0.46, 0.62)
    lms[10] = MockLandmark(0.50, 0.54)
    lms[12] = MockLandmark(0.50, 0.62)
    lms[13] = MockLandmark(0.54, 0.52)
    lms[14] = MockLandmark(0.54, 0.56)
    lms[16] = MockLandmark(0.54, 0.62)
    lms[17] = MockLandmark(0.58, 0.54)
    lms[18] = MockLandmark(0.58, 0.58)
    lms[20] = MockLandmark(0.58, 0.64)
    # Thumb pointing straight UP high above knuckles
    lms[1] = MockLandmark(0.46, 0.70)
    lms[2] = MockLandmark(0.44, 0.58)
    lms[3] = MockLandmark(0.44, 0.46)
    lms[4] = MockLandmark(0.44, 0.32)

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms), phrase_mode=True)
    assert pred == "YES"
    assert conf >= 0.95


def test_classify_thumbs_down_no_phrase_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.70)  # wrist
    lms[9] = MockLandmark(0.5, 0.45)  # mid MCP
    # All 4 fingers curled into fist
    lms[5] = MockLandmark(0.46, 0.48)
    lms[6] = MockLandmark(0.46, 0.52)
    lms[8] = MockLandmark(0.46, 0.58)
    lms[10] = MockLandmark(0.50, 0.52)
    lms[12] = MockLandmark(0.50, 0.58)
    lms[13] = MockLandmark(0.54, 0.50)
    lms[14] = MockLandmark(0.54, 0.54)
    lms[16] = MockLandmark(0.54, 0.58)
    lms[17] = MockLandmark(0.58, 0.52)
    lms[18] = MockLandmark(0.58, 0.56)
    lms[20] = MockLandmark(0.58, 0.60)
    # Thumb pointing straight DOWN below wrist
    lms[1] = MockLandmark(0.46, 0.65)
    lms[2] = MockLandmark(0.45, 0.72)
    lms[3] = MockLandmark(0.45, 0.85)
    lms[4] = MockLandmark(0.45, 0.96)

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms), phrase_mode=True)
    assert pred == "NO"
    assert conf >= 0.95


def test_classify_okay_phrase_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)   # wrist
    lms[9] = MockLandmark(0.5, 0.5)   # mid MCP
    # Index tip touching thumb tip
    lms[5] = MockLandmark(0.45, 0.50)
    lms[6] = MockLandmark(0.44, 0.45)
    lms[8] = MockLandmark(0.42, 0.40)
    lms[1] = MockLandmark(0.46, 0.65)
    lms[2] = MockLandmark(0.44, 0.58)
    lms[4] = MockLandmark(0.42, 0.40) # touches index tip
    # Middle, Ring, Pinky extended UP
    lms[10] = MockLandmark(0.50, 0.38)
    lms[12] = MockLandmark(0.50, 0.20)
    lms[13] = MockLandmark(0.55, 0.52)
    lms[14] = MockLandmark(0.55, 0.39)
    lms[16] = MockLandmark(0.55, 0.22)
    lms[17] = MockLandmark(0.60, 0.54)
    lms[18] = MockLandmark(0.60, 0.41)
    lms[20] = MockLandmark(0.60, 0.25)

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms), phrase_mode=True)
    assert pred == "OKAY"
    assert conf >= 0.95


def test_classify_stop_phrase_sign():
    clf = LandmarkSignClassifier()
    lms = [MockLandmark(0.5, 0.8)] * 21
    lms[0] = MockLandmark(0.5, 0.8)   # wrist
    lms[9] = MockLandmark(0.5, 0.5)   # mid MCP
    # All fingers extended UP
    lms[5] = MockLandmark(0.44, 0.50)
    lms[6] = MockLandmark(0.44, 0.35)
    lms[8] = MockLandmark(0.44, 0.15)
    lms[10] = MockLandmark(0.50, 0.35)
    lms[12] = MockLandmark(0.50, 0.15)
    lms[13] = MockLandmark(0.55, 0.52)
    lms[14] = MockLandmark(0.55, 0.36)
    lms[16] = MockLandmark(0.55, 0.17)
    lms[17] = MockLandmark(0.60, 0.54)
    lms[18] = MockLandmark(0.60, 0.38)
    lms[20] = MockLandmark(0.60, 0.20)
    # Thumb abducted
    lms[1] = MockLandmark(0.48, 0.70)
    lms[2] = MockLandmark(0.42, 0.62)
    lms[4] = MockLandmark(0.32, 0.50)

    pred, conf, cands = clf.classify_landmarks(MockLandmarkList(lms), phrase_mode=True)
    assert pred == "STOP"
    assert conf >= 0.95




