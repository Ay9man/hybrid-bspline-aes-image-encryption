# ============================================================
# Hybrid 2D B-Spline + AES-CBC Image Encryption
# ============================================================
#
# Recommended Colab install command:
#
# IMPORTANT:
# 1) AES key and chaotic control material are derived from a 256-bit master key using HKDF-SHA256.
# 2) AES-CBC uses a fresh random 128-bit IV in normal operation.
# 3) Logistic map controls ONLY the B-spline deformation; it does not provide AES key entropy.
# 4) The B-spline surface generates a key-dependent DISCRETE pixel permutation.
# 5) No interpolation is used in the confusion/deconfusion stage; reconstruction is exact.
# 6) A reversible NONLINEAR bidirectional key-dependent diffusion stage follows B-spline confusion.
# 7) NPCR/UACI subtraction is performed in signed integer arithmetic to avoid uint8 overflow.
# 8) Differential tests use the SAME controlled IV for each plaintext pair.
# 9) AES-CBC provides confidentiality, not authentication.
#
# ============================================================

import os
import glob
import time
import math
import secrets
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import kagglehub

from scipy.interpolate import RectBivariateSpline
from scipy.stats import chisquare, norm
from skimage.measure import shannon_entropy
from skimage import metrics

from Crypto.Cipher import AES
from Crypto.Protocol.KDF import HKDF
from Crypto.Hash import SHA256, HMAC
from Crypto.Random import get_random_bytes

# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 256                   # Matches the ORIGINAL implementation.
NUM_IMAGES = 10                    # Use None to process all available test images.
R_LOGISTIC = 3.99                  # Public system parameter.
LOGISTIC_BURN_IN = 1000
HKDF_SALT = b"IJEER-Hybrid-BSpline-AES-v1"  # Public salt.

# 2D B-spline control grid.
CONTROL_GRID_Y = 8
CONTROL_GRID_X = 8

# Initial maximum control-point displacement in pixels.
# The code automatically scales this down if needed to prevent foldovers.
ALPHA_X = 6.0
BETA_Y = 6.0
MIN_JACOBIAN = 0.20
BOUNDARY_TOL = 1e-6              # Numerical tolerance for spline edge round-off.

# Fixed-point iterations for numerical inverse deformation.
INVERSE_ITERATIONS = 20
INVERSE_TOL = 1e-4

# Differential testing.
DIFF_TRIALS = 100
DIFF_ALPHA = 0.05

# Key-sensitivity bit positions.
KEY_BIT_POSITIONS = [0, 63, 127, 191, 255]

# Set this to a 64-hex-character value if you need exact reproducibility.
# Example:
# MASTER_KEY_HEX = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
MASTER_KEY_HEX = None

# ============================================================
# Dataset
# ============================================================

path = kagglehub.dataset_download(
    "balraj98/berkeley-segmentation-dataset-500-bsds500"
)
print(f"Dataset downloaded to: {path}")

image_dir = os.path.join(path, "images", "test")
image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

if not image_files:
    raise FileNotFoundError(
        f"No image files found in {image_dir}. "
        "Please verify the downloaded BSDS500 structure."
    )

if NUM_IMAGES is not None:
    image_files = image_files[:NUM_IMAGES]

print(f"Number of images selected: {len(image_files)}")
print("Selected files:")
for f in image_files:
    print("  ", os.path.basename(f))

# ============================================================
# Master key
# ============================================================

if MASTER_KEY_HEX is None:
    MASTER_KEY = get_random_bytes(32)
    print(
        "\nA fresh 256-bit master key was generated for this run. "
        "Save it securely if exact reproducibility is required."
    )
else:
    MASTER_KEY = bytes.fromhex(MASTER_KEY_HEX)
    if len(MASTER_KEY) != 32:
        raise ValueError("MASTER_KEY_HEX must encode exactly 32 bytes (256 bits).")

master_key_fingerprint = SHA256.new(MASTER_KEY).hexdigest()[:16]
print(f"Master-key fingerprint (not the key): {master_key_fingerprint}")
print("Pipeline: lossless B-spline-guided permutation -> nonlinear bidirectional diffusion -> AES-CBC")

# ============================================================
# Key derivation and chaotic sequence generation
# ============================================================

def derive_key_material(master_key, salt=HKDF_SALT):
    """
    Domain-separated HKDF-SHA256 derivation.

    Returns:
        aes_key   : 16 bytes (AES-128)
        chaos_key : 32 bytes (Logistic-map initial-state material)
        diff_key  : 32 bytes (bidirectional preprocessing diffusion)
    """
    aes_key = HKDF(
        master=master_key,
        key_len=16,
        salt=salt,
        hashmod=SHA256,
        context=b"AES-KEY"
    )

    chaos_key = HKDF(
        master=master_key,
        key_len=32,
        salt=salt,
        hashmod=SHA256,
        context=b"B-SPLINE-CHAOS"
    )

    diff_key = HKDF(
        master=master_key,
        key_len=32,
        salt=salt,
        hashmod=SHA256,
        context=b"DIFFUSION-KEY"
    )

    return aes_key, chaos_key, diff_key

def chaos_key_to_x0(chaos_key, eps=1e-12):
    """
    Maps the 256-bit derived chaos material into (eps, 1-eps).
    Logistic-map arithmetic is still floating-point; therefore the chaotic
    state is NOT counted as independent cryptographic entropy.
    """
    z = int.from_bytes(chaos_key, byteorder="big", signed=False)
    u = z / float((1 << 256) - 1)
    return eps + (1.0 - 2.0 * eps) * u


def logistic_sequence(x0, count, r=R_LOGISTIC, burn_in=LOGISTIC_BURN_IN):
    """
    Generates deterministic Logistic-map samples after a transient burn-in.
    """
    x = float(x0)

    for _ in range(burn_in):
        x = r * x * (1.0 - x)

    out = np.empty(count, dtype=np.float64)

    for i in range(count):
        x = r * x * (1.0 - x)
        out[i] = x

    return out


# ============================================================
# 2D B-spline deformation
# ============================================================

def _interpolate_control_field(ctrl, height, width):
    """
    Cubic B-spline interpolation of a coarse control field to full resolution.
    """
    gy, gx = ctrl.shape

    cy = np.linspace(0, height - 1, gy)
    cx = np.linspace(0, width - 1, gx)

    ky = min(3, gy - 1)
    kx = min(3, gx - 1)

    # RectBivariateSpline first coordinate corresponds to rows (y),
    # second coordinate to columns (x).
    spline = RectBivariateSpline(cy, cx, ctrl, kx=ky, ky=kx)

    yy = np.arange(height, dtype=np.float64)
    xx = np.arange(width, dtype=np.float64)

    return spline(yy, xx)


def _jacobian_determinant(map_x, map_y):
    """
    Computes det(J_F) for F(x,y) = (map_x, map_y).
    """
    dmapx_dy, dmapx_dx = np.gradient(map_x)
    dmapy_dy, dmapy_dx = np.gradient(map_y)

    return dmapx_dx * dmapy_dy - dmapx_dy * dmapy_dx


def build_bspline_deformation(
    height,
    width,
    chaos_key,
    grid_y=CONTROL_GRID_Y,
    grid_x=CONTROL_GRID_X,
    alpha_x=ALPHA_X,
    beta_y=BETA_Y,
    min_jacobian=MIN_JACOBIAN
):
    """
    Builds a key-dependent 2D B-spline displacement field.

    Boundary control points are fixed to zero displacement.
    The displacement magnitude is automatically reduced until:
        1) the forward map stays inside the image domain, and
        2) det(J_F) remains above min_jacobian.

    Returns a dictionary containing full-resolution deformation maps.
    """
    x0 = chaos_key_to_x0(chaos_key)

    n_ctrl = grid_y * grid_x
    seq = logistic_sequence(x0, 2 * n_ctrl)

    ctrl_dx = alpha_x * (2.0 * seq[:n_ctrl] - 1.0)
    ctrl_dy = beta_y * (2.0 * seq[n_ctrl:] - 1.0)

    ctrl_dx = ctrl_dx.reshape(grid_y, grid_x)
    ctrl_dy = ctrl_dy.reshape(grid_y, grid_x)

    # Anchor all boundaries to avoid uncontrolled edge displacement.
    ctrl_dx[0, :] = 0
    ctrl_dx[-1, :] = 0
    ctrl_dx[:, 0] = 0
    ctrl_dx[:, -1] = 0

    ctrl_dy[0, :] = 0
    ctrl_dy[-1, :] = 0
    ctrl_dy[:, 0] = 0
    ctrl_dy[:, -1] = 0

    base_dx = _interpolate_control_field(ctrl_dx, height, width)
    base_dy = _interpolate_control_field(ctrl_dy, height, width)

    # RectBivariateSpline may introduce tiny floating-point residuals at
    # anchored boundaries (for example -1e-16 instead of exactly 0).
    # Force the full-resolution outer boundary displacement to exactly zero.
    base_dx[0, :] = 0.0
    base_dx[-1, :] = 0.0
    base_dx[:, 0] = 0.0
    base_dx[:, -1] = 0.0

    base_dy[0, :] = 0.0
    base_dy[-1, :] = 0.0
    base_dy[:, 0] = 0.0
    base_dy[:, -1] = 0.0

    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij"
    )

    scale = 1.0
    accepted = False

    for attempt in range(80):
        dx = scale * base_dx
        dy = scale * base_dy

        forward_x_raw = xx + dx
        forward_y_raw = yy + dy

        # Evaluate the Jacobian BEFORE any numerical clipping.
        jac = _jacobian_determinant(forward_x_raw, forward_y_raw)

        # Allow only a tiny floating-point tolerance at image boundaries.
        within_bounds = (
            forward_x_raw.min() >= -BOUNDARY_TOL
            and forward_x_raw.max() <= (width - 1) + BOUNDARY_TOL
            and forward_y_raw.min() >= -BOUNDARY_TOL
            and forward_y_raw.max() <= (height - 1) + BOUNDARY_TOL
        )

        jacobian_ok = np.isfinite(jac).all() and np.min(jac) > min_jacobian

        if within_bounds and jacobian_ok:
            accepted = True
            break

        scale *= 0.85

    if not accepted:
        raise RuntimeError(
            "Could not construct a stable fold-free B-spline deformation. "
            f"Last scale={scale:.6g}, "
            f"min Jacobian={np.nanmin(jac):.6g}, "
            f"x-range=[{np.nanmin(forward_x_raw):.6g}, {np.nanmax(forward_x_raw):.6g}], "
            f"y-range=[{np.nanmin(forward_y_raw):.6g}, {np.nanmax(forward_y_raw):.6g}]."
        )

    # Remove only numerical edge excursions after the mapping has passed
    # the Jacobian and bounds tests.
    forward_x = np.clip(forward_x_raw, 0.0, width - 1.0)
    forward_y = np.clip(forward_y_raw, 0.0, height - 1.0)

    # Solve q = p + D(p) for p using fixed-point iteration.
    # q is every target coordinate; p is source coordinate.
    inv_x = xx.copy().astype(np.float32)
    inv_y = yy.copy().astype(np.float32)

    dx32 = dx.astype(np.float32)
    dy32 = dy.astype(np.float32)
    xx32 = xx.astype(np.float32)
    yy32 = yy.astype(np.float32)

    for _ in range(INVERSE_ITERATIONS):
        sampled_dx = cv2.remap(
            dx32,
            inv_x,
            inv_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        sampled_dy = cv2.remap(
            dy32,
            inv_x,
            inv_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        new_inv_x = xx32 - sampled_dx
        new_inv_y = yy32 - sampled_dy

        error = max(
            float(np.max(np.abs(new_inv_x - inv_x))),
            float(np.max(np.abs(new_inv_y - inv_y)))
        )

        inv_x = new_inv_x
        inv_y = new_inv_y

        if error < INVERSE_TOL:
            break

    return {
        "x0": x0,
        "scale": scale,
        "dx": dx.astype(np.float32),
        "dy": dy.astype(np.float32),
        "forward_x": forward_x.astype(np.float32),
        "forward_y": forward_y.astype(np.float32),
        "inverse_x": inv_x.astype(np.float32),
        "inverse_y": inv_y.astype(np.float32),
        "jacobian": jac.astype(np.float64),
        "min_jacobian": float(np.min(jac)),
        "max_jacobian": float(np.max(jac)),
    }


def build_bspline_permutation(deformation):
    """
    Converts the smooth key-dependent B-spline deformation into a discrete,
    exactly invertible pixel permutation.

    Each original pixel has a continuous warped coordinate:
        q = F(p) = (x + dx, y + dy)

    Pixels are ranked lexicographically by their warped (y', x') coordinates.
    The ranking creates a one-to-one permutation of ALL discrete pixel
    positions. No interpolation, averaging, rounding collision, or pixel loss
    occurs.

    A stable original-index tie breaker is included for numerical robustness.
    """
    warped_x = deformation["forward_x"].ravel().astype(np.float64)
    warped_y = deformation["forward_y"].ravel().astype(np.float64)
    original_index = np.arange(warped_x.size, dtype=np.int64)

    # np.lexsort uses the LAST key as the primary key:
    # primary: warped_y, secondary: warped_x, final tie-break: original index.
    perm = np.lexsort((original_index, warped_x, warped_y)).astype(np.int64)

    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(perm.size, dtype=np.int64)

    return perm, inv_perm


def bspline_confuse(image, deformation):
    """
    Exact B-spline-guided discrete confusion.

    The B-spline deformation determines the ordering, but the actual image
    operation is a pure permutation:
        I_c[k] = I[perm[k]]

    Therefore every input byte appears exactly once in the confused image.
    """
    perm = deformation.get("perm")
    if perm is None:
        perm, _ = build_bspline_permutation(deformation)
    confused = image.ravel()[perm]
    return confused.reshape(image.shape).copy()


def bspline_reconstruct(confused_image, deformation):
    """
    Exact inverse of the B-spline-guided discrete permutation.

    If:
        confused[k] = original[perm[k]]
    then:
        original[perm[k]] = confused[k]
    """
    perm = deformation.get("perm")
    if perm is None:
        perm, _ = build_bspline_permutation(deformation)

    recovered = np.empty(confused_image.size, dtype=np.uint8)
    recovered[perm] = confused_image.ravel()

    return recovered.reshape(confused_image.shape)


def verify_bspline_permutation_inverse():
    """
    Sanity check for the permutation logic using an artificial deformation.
    The real keyed deformation is additionally checked during each run.
    """
    h = w = 16
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij"
    )

    # Small deterministic, fold-free synthetic displacement.
    fx = xx + 0.15 * np.sin(yy / 3.0)
    fy = yy + 0.15 * np.cos(xx / 4.0)

    deformation = {
        "forward_x": fx,
        "forward_y": fy
    }

    test = np.arange(h * w, dtype=np.uint8).reshape(h, w)
    confused = bspline_confuse(test, deformation)
    recovered = bspline_reconstruct(confused, deformation)

    if not np.array_equal(test, recovered):
        raise RuntimeError("B-spline permutation inverse self-test failed.")


verify_bspline_permutation_inverse()


# ============================================================
# Bidirectional key-dependent preprocessing diffusion
# ============================================================

def hmac_sha256_keystream(key, length, label):
    """
    Expands a 256-bit diffusion key into an arbitrary-length deterministic
    byte stream using HMAC-SHA256 in counter mode.

    The forward and reverse passes use distinct labels for domain separation.
    This keystream generator is used only for the reversible preprocessing
    diffusion layer; AES-CBC remains the confidentiality primitive.
    """
    out = bytearray()
    counter = 0

    while len(out) < length:
        msg = label + counter.to_bytes(8, byteorder="big", signed=False)
        h = HMAC.new(key, digestmod=SHA256)
        h.update(msg)
        out.extend(h.digest())
        counter += 1

    return np.frombuffer(bytes(out[:length]), dtype=np.uint8).copy()



# AES S-box and inverse S-box are used here only as public bijective
# nonlinear lookup tables inside the reversible preprocessing diffusion.
AES_SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
], dtype=np.uint8)

AES_INV_SBOX = np.empty(256, dtype=np.uint8)
AES_INV_SBOX[AES_SBOX] = np.arange(256, dtype=np.uint8)

def bidirectional_diffuse(image, diff_key):
    """
    Reversible nonlinear two-pass byte diffusion.

    Let B be the serialized B-spline-permuted image and let Rf and Rr be
    independent HMAC-SHA256-derived byte streams.

    Forward pass:
        F[0] = S(B[0] XOR Rf[0])
        F[i] = S(B[i] XOR Rf[i] XOR F[i-1])

    Reverse pass:
        D[N-1] = S(F[N-1] XOR Rr[N-1])
        D[i] = S(F[i] XOR Rr[i] XOR D[i+1])

    S(.) is the AES S-box used as a public nonlinear bijection. The layer is
    exactly invertible because S has a known inverse permutation.

    This stage is preprocessing only; AES-CBC remains the confidentiality
    primitive of the framework.
    """
    if image.dtype != np.uint8:
        raise TypeError("Diffusion input image must be uint8.")

    shape = image.shape
    b = image.ravel()
    n = b.size

    rf = hmac_sha256_keystream(diff_key, n, b"DIFF-FORWARD")
    rr = hmac_sha256_keystream(diff_key, n, b"DIFF-REVERSE")

    f = np.empty(n, dtype=np.uint8)
    d = np.empty(n, dtype=np.uint8)

    f[0] = AES_SBOX[int(b[0] ^ rf[0])]
    for i in range(1, n):
        f[i] = AES_SBOX[int(b[i] ^ rf[i] ^ f[i - 1])]

    d[-1] = AES_SBOX[int(f[-1] ^ rr[-1])]
    for i in range(n - 2, -1, -1):
        d[i] = AES_SBOX[int(f[i] ^ rr[i] ^ d[i + 1])]

    return d.reshape(shape).copy()

def inverse_bidirectional_diffuse(diffused_image, diff_key):
    """
    Exact inverse of the nonlinear bidirectional diffusion.
    """
    if diffused_image.dtype != np.uint8:
        raise TypeError("Inverse-diffusion input must be uint8.")

    shape = diffused_image.shape
    d = diffused_image.ravel()
    n = d.size

    rf = hmac_sha256_keystream(diff_key, n, b"DIFF-FORWARD")
    rr = hmac_sha256_keystream(diff_key, n, b"DIFF-REVERSE")

    f = np.empty(n, dtype=np.uint8)
    b = np.empty(n, dtype=np.uint8)

    # Invert reverse pass.
    f[-1] = AES_INV_SBOX[int(d[-1])] ^ rr[-1]
    for i in range(n - 2, -1, -1):
        f[i] = AES_INV_SBOX[int(d[i])] ^ rr[i] ^ d[i + 1]

    # Invert forward pass.
    b[0] = AES_INV_SBOX[int(f[0])] ^ rf[0]
    for i in range(1, n):
        b[i] = AES_INV_SBOX[int(f[i])] ^ rf[i] ^ f[i - 1]

    return b.reshape(shape).copy()

def verify_diffusion_inverse():
    """
    Internal sanity check: diffusion followed by inverse diffusion must be
    exactly lossless at the byte level.
    """
    test_key = bytes(range(32))
    test = np.arange(256, dtype=np.uint8).reshape(16, 16)
    diff = bidirectional_diffuse(test, test_key)
    rec = inverse_bidirectional_diffuse(diff, test_key)

    if not np.array_equal(test, rec):
        raise RuntimeError("Bidirectional diffusion inverse self-test failed.")


verify_diffusion_inverse()

# ============================================================
# AES-CBC
# ============================================================

def aes_cbc_encrypt(image, aes_key, iv=None):
    """
    AES-CBC encryption of an 8-bit grayscale image.

    For 256x256 images, byte length = 65536, which is exactly divisible by 16,
    so no padding is required.
    """
    if image.dtype != np.uint8:
        raise TypeError("AES input image must be uint8.")

    plaintext = image.tobytes()

    if len(plaintext) % AES.block_size != 0:
        raise ValueError(
            "Image byte length is not AES-block aligned. "
            "For the current paper experiments use 256x256 uint8 grayscale images."
        )

    if iv is None:
        iv = get_random_bytes(AES.block_size)

    if len(iv) != AES.block_size:
        raise ValueError("AES-CBC IV must be exactly 16 bytes.")

    cipher = AES.new(aes_key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(plaintext)

    encrypted_img = np.frombuffer(ciphertext, dtype=np.uint8).reshape(image.shape)

    return ciphertext, encrypted_img, iv


def aes_cbc_decrypt(ciphertext, shape, aes_key, iv):
    cipher = AES.new(aes_key, AES.MODE_CBC, iv)
    plaintext = cipher.decrypt(ciphertext)

    expected_size = int(np.prod(shape))
    if len(plaintext) != expected_size:
        raise ValueError("Unexpected decrypted byte length.")

    return np.frombuffer(plaintext, dtype=np.uint8).reshape(shape).copy()


# ============================================================
# Complete hybrid pipeline
# ============================================================

def prepare_keyed_system(master_key, shape):
    aes_key, chaos_key, diff_key = derive_key_material(master_key)
    deformation = build_bspline_deformation(
        height=shape[0],
        width=shape[1],
        chaos_key=chaos_key
    )

    perm, inv_perm = build_bspline_permutation(deformation)

    if perm.size != shape[0] * shape[1]:
        raise RuntimeError("Invalid B-spline permutation length.")

    if np.unique(perm).size != perm.size:
        raise RuntimeError("B-spline permutation is not one-to-one.")

    # Cache permutation arrays so repeated trials do not sort again.
    deformation["perm"] = perm
    deformation["inv_perm"] = inv_perm

    return aes_key, chaos_key, diff_key, deformation


def hybrid_encrypt(image, master_key, iv=None, prepared=None):
    if prepared is None:
        aes_key, chaos_key, diff_key, deformation = prepare_keyed_system(
            master_key, image.shape
        )
    else:
        aes_key, chaos_key, diff_key, deformation = prepared

    t0 = time.perf_counter()
    confused = bspline_confuse(image, deformation)
    t1 = time.perf_counter()

    diffused = bidirectional_diffuse(confused, diff_key)
    t2 = time.perf_counter()

    ciphertext, encrypted_img, iv = aes_cbc_encrypt(
        diffused, aes_key, iv=iv
    )
    t3 = time.perf_counter()

    return {
        "ciphertext": ciphertext,
        "encrypted_image": encrypted_img,
        "iv": iv,
        "confused_image": confused,
        "diffused_image": diffused,
        "deformation": deformation,
        "aes_key": aes_key,
        "chaos_key": chaos_key,
        "diff_key": diff_key,
        "time_bspline": t1 - t0,
        "time_diffusion": t2 - t1,
        "time_aes": t3 - t2,
        "time_total": t3 - t0
    }

def hybrid_decrypt(ciphertext, shape, master_key, iv, prepared=None):
    if prepared is None:
        aes_key, chaos_key, diff_key, deformation = prepare_keyed_system(
            master_key, shape
        )
    else:
        aes_key, chaos_key, diff_key, deformation = prepared

    t0 = time.perf_counter()
    decrypted_diffused = aes_cbc_decrypt(
        ciphertext, shape, aes_key, iv
    )
    t1 = time.perf_counter()

    decrypted_confused = inverse_bidirectional_diffuse(
        decrypted_diffused, diff_key
    )
    t2 = time.perf_counter()

    recovered = bspline_reconstruct(
        decrypted_confused, deformation
    )
    t3 = time.perf_counter()

    return {
        "decrypted_diffused": decrypted_diffused,
        "decrypted_confused": decrypted_confused,
        "recovered_image": recovered,
        "time_aes": t1 - t0,
        "time_diffusion": t2 - t1,
        "time_bspline": t3 - t2,
        "time_total": t3 - t0
    }


# ============================================================
# Metrics
# ============================================================

def entropy(image):
    return float(shannon_entropy(image))


def adjacent_correlations(image):
    """
    Returns horizontal, vertical, and diagonal adjacent-pixel correlation.
    """
    im = image.astype(np.float64)

    horizontal = np.corrcoef(
        im[:, :-1].ravel(),
        im[:, 1:].ravel()
    )[0, 1]

    vertical = np.corrcoef(
        im[:-1, :].ravel(),
        im[1:, :].ravel()
    )[0, 1]

    diagonal = np.corrcoef(
        im[:-1, :-1].ravel(),
        im[1:, 1:].ravel()
    )[0, 1]

    return {
        "H": float(horizontal),
        "V": float(vertical),
        "D": float(diagonal)
    }


def calculate_histogram_variance(image):
    hist, _ = np.histogram(
        image.ravel(),
        bins=256,
        range=(0, 256)
    )
    return float(np.var(hist))


def perform_chi_square_test(image):
    observed, _ = np.histogram(
        image.ravel(),
        bins=256,
        range=(0, 256)
    )

    expected = np.full(
        256,
        image.size / 256.0,
        dtype=np.float64
    )

    chi2_stat, p_value = chisquare(
        f_obs=observed,
        f_exp=expected
    )

    return float(chi2_stat), float(p_value)


def reconstruction_metrics(original, recovered):
    a = original.astype(np.float64)
    b = recovered.astype(np.float64)

    diff = a - b
    mse = float(np.mean(diff ** 2))

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(
            metrics.peak_signal_noise_ratio(
                original,
                recovered,
                data_range=255
            )
        )

    ssim = float(
        metrics.structural_similarity(
            original,
            recovered,
            data_range=255
        )
    )

    max_abs_error = int(np.max(np.abs(diff)))
    pmr = float(np.mean(original != recovered) * 100.0)

    return {
        "MSE": mse,
        "PSNR": psnr,
        "SSIM": ssim,
        "MaxAbsError": max_abs_error,
        "PMR": pmr
    }


def npcr_uaci(cipher_a, cipher_b):
    """
    Correct NPCR/UACI.

    CRITICAL FIX:
    Cast to signed integer BEFORE subtraction.
    The old code used:
        np.abs(uint8_a - uint8_b)
    which wraps modulo 256 and artificially drives UACI toward ~50%.
    """
    a = np.asarray(cipher_a, dtype=np.uint8)
    b = np.asarray(cipher_b, dtype=np.uint8)

    if a.shape != b.shape:
        raise ValueError("Cipher arrays must have identical shapes.")

    npcr = float(np.mean(a != b) * 100.0)

    a_signed = a.astype(np.int16)
    b_signed = b.astype(np.int16)

    uaci = float(
        np.mean(np.abs(a_signed - b_signed)) / 255.0 * 100.0
    )

    return npcr, uaci


def differential_randomness_thresholds(
    height,
    width,
    alpha=DIFF_ALPHA
):
    """
    Size-aware normal-approximation thresholds.

    NPCR:
      one-sided lower critical value.

    UACI:
      two-sided acceptance interval based on the exact first two moments of
      |X-Y|/255 for independent uniform 8-bit values.

    This avoids hard-coding 512x512 thresholds when experiments are 256x256.
    """
    n = height * width

    # NPCR under independent uniform bytes.
    p = 255.0 / 256.0
    z_one = norm.ppf(1.0 - alpha)
    npcr_lower = (
        p - z_one * math.sqrt(p * (1.0 - p) / n)
    ) * 100.0

    # Exact single-pixel UACI distribution moments.
    values = np.arange(256, dtype=np.float64)
    normalized_diff = (
        np.abs(values[:, None] - values[None, :]) / 255.0
    )

    mu = float(np.mean(normalized_diff))
    sigma = float(np.std(normalized_diff))

    z_two = norm.ppf(1.0 - alpha / 2.0)
    se = sigma / math.sqrt(n)

    uaci_low = (mu - z_two * se) * 100.0
    uaci_high = (mu + z_two * se) * 100.0

    return {
        "NPCR_expected": p * 100.0,
        "NPCR_lower": npcr_lower,
        "UACI_expected": mu * 100.0,
        "UACI_low": uaci_low,
        "UACI_high": uaci_high,
        "alpha": alpha
    }


# ============================================================
# Differential plaintext-sensitivity test
# ============================================================

def change_one_pixel_by_one_level(image, rng):
    modified = image.copy()

    i = int(rng.integers(0, image.shape[0]))
    j = int(rng.integers(0, image.shape[1]))

    if modified[i, j] < 255:
        modified[i, j] = np.uint8(modified[i, j] + 1)
    else:
        modified[i, j] = np.uint8(modified[i, j] - 1)

    return modified, (i, j)


def differential_test(
    image,
    master_key,
    trials=DIFF_TRIALS,
    alpha=DIFF_ALPHA,
    rng_seed=20260813
):
    """
    Controlled differential test:
      - same master key
      - same IV
      - only one plaintext pixel changes by one intensity level
    """
    prepared = prepare_keyed_system(master_key, image.shape)

    test_iv = get_random_bytes(16)

    baseline = hybrid_encrypt(
        image,
        master_key,
        iv=test_iv,
        prepared=prepared
    )

    baseline_cipher = baseline["encrypted_image"]

    rng = np.random.default_rng(rng_seed)

    rows = []

    for t in range(trials):
        modified, pos = change_one_pixel_by_one_level(image, rng)

        result = hybrid_encrypt(
            modified,
            master_key,
            iv=test_iv,
            prepared=prepared
        )

        npcr, uaci = npcr_uaci(
            baseline_cipher,
            result["encrypted_image"]
        )

        rows.append({
            "Trial": t + 1,
            "PixelRow": pos[0],
            "PixelCol": pos[1],
            "NPCR": npcr,
            "UACI": uaci
        })

    df = pd.DataFrame(rows)
    th = differential_randomness_thresholds(
        image.shape[0],
        image.shape[1],
        alpha=alpha
    )

    df["NPCR_Pass"] = df["NPCR"] >= th["NPCR_lower"]
    df["UACI_Pass"] = (
        (df["UACI"] >= th["UACI_low"])
        & (df["UACI"] <= th["UACI_high"])
    )

    summary = {
        "NPCR_Mean": float(df["NPCR"].mean()),
        "NPCR_SD": float(df["NPCR"].std(ddof=1)),
        "NPCR_PassRate": float(df["NPCR_Pass"].mean() * 100.0),
        "UACI_Mean": float(df["UACI"].mean()),
        "UACI_SD": float(df["UACI"].std(ddof=1)),
        "UACI_PassRate": float(df["UACI_Pass"].mean() * 100.0),
        **th
    }

    return df, summary


# ============================================================
# Key-sensitivity test
# ============================================================

def flip_master_key_bit(master_key, bit_position):
    if not (0 <= bit_position <= 255):
        raise ValueError("bit_position must be between 0 and 255.")

    value = int.from_bytes(master_key, "big")
    modified = value ^ (1 << bit_position)
    return modified.to_bytes(32, "big")


def key_sensitivity_test(
    image,
    master_key,
    bit_positions=KEY_BIT_POSITIONS
):
    """
    Uses the same controlled IV so ciphertext differences are caused by the
    one-bit master-key change rather than by IV variation.
    """
    test_iv = get_random_bytes(16)

    prepared_correct = prepare_keyed_system(master_key, image.shape)

    baseline = hybrid_encrypt(
        image,
        master_key,
        iv=test_iv,
        prepared=prepared_correct
    )

    correct_dec = hybrid_decrypt(
        baseline["ciphertext"],
        image.shape,
        master_key,
        test_iv,
        prepared=prepared_correct
    )

    correct_metrics = reconstruction_metrics(
        image,
        correct_dec["recovered_image"]
    )

    rows = []

    for bit in bit_positions:
        wrong_key = flip_master_key_bit(master_key, bit)

        prepared_wrong = prepare_keyed_system(
            wrong_key,
            image.shape
        )

        wrong_encryption = hybrid_encrypt(
            image,
            wrong_key,
            iv=test_iv,
            prepared=prepared_wrong
        )

        npcr, uaci = npcr_uaci(
            baseline["encrypted_image"],
            wrong_encryption["encrypted_image"]
        )

        wrong_dec = hybrid_decrypt(
            baseline["ciphertext"],
            image.shape,
            wrong_key,
            test_iv,
            prepared=prepared_wrong
        )

        wrong_metrics = reconstruction_metrics(
            image,
            wrong_dec["recovered_image"]
        )

        rows.append({
            "BitPosition": bit,
            "Cipher_NPCR": npcr,
            "Cipher_UACI": uaci,
            "WrongKey_MSE": wrong_metrics["MSE"],
            "WrongKey_PSNR": wrong_metrics["PSNR"],
            "WrongKey_SSIM": wrong_metrics["SSIM"],
            "WrongKey_PMR": wrong_metrics["PMR"],
            "CorrectKey_PSNR": correct_metrics["PSNR"],
            "CorrectKey_SSIM": correct_metrics["SSIM"]
        })

    df = pd.DataFrame(rows)

    summary = {
        "Key_NPCR_Mean": float(df["Cipher_NPCR"].mean()),
        "Key_NPCR_SD": float(df["Cipher_NPCR"].std(ddof=1)),
        "Key_UACI_Mean": float(df["Cipher_UACI"].mean()),
        "Key_UACI_SD": float(df["Cipher_UACI"].std(ddof=1)),
        "WrongKey_PSNR_Mean": float(df["WrongKey_PSNR"].mean()),
        "WrongKey_SSIM_Mean": float(df["WrongKey_SSIM"].mean()),
        "CorrectKey_PSNR": correct_metrics["PSNR"],
        "CorrectKey_SSIM": correct_metrics["SSIM"]
    }

    return df, summary


# ============================================================
# Optional ablation analysis
# ============================================================

def aes_only_encrypt(image, master_key, iv):
    aes_key, _, _ = derive_key_material(master_key)
    return aes_cbc_encrypt(image, aes_key, iv=iv)


def ablation_test(image, master_key):
    """
    Compares:
      1) B-spline only
      2) AES-CBC only
      3) lossless B-spline permutation + bidirectional diffusion (preprocessing only)
      4) Full hybrid: B-spline + diffusion + AES-CBC

    The preprocessing-only rows are not presented as standalone secure ciphers;
    they quantify the contribution of each stage.
    """
    prepared = prepare_keyed_system(master_key, image.shape)
    aes_key, _, diff_key, deformation = prepared

    common_iv = get_random_bytes(16)

    # B-spline only.
    t0 = time.perf_counter()
    confused = bspline_confuse(image, deformation)
    t1 = time.perf_counter()

    # AES-CBC only.
    t2 = time.perf_counter()
    _, aes_img, _ = aes_cbc_encrypt(
        image,
        aes_key,
        iv=common_iv
    )
    t3 = time.perf_counter()

    # B-spline + diffusion preprocessing.
    t4 = time.perf_counter()
    preprocessed = bidirectional_diffuse(confused, diff_key)
    t5 = time.perf_counter()

    # Full hybrid.
    hybrid = hybrid_encrypt(
        image,
        master_key,
        iv=common_iv,
        prepared=prepared
    )

    rows = []

    for name, out_img, elapsed in [
        ("B-spline only", confused, t1 - t0),
        ("AES-CBC only", aes_img, t3 - t2),
        ("B-spline + nonlinear diffusion", preprocessed, (t1 - t0) + (t5 - t4)),
        ("Full hybrid", hybrid["encrypted_image"], hybrid["time_total"])
    ]:
        corr = adjacent_correlations(out_img)
        rows.append({
            "Method": name,
            "Entropy": entropy(out_img),
            "Correlation_H": corr["H"],
            "Correlation_V": corr["V"],
            "Correlation_D": corr["D"],
            "Time_s": elapsed
        })

    return pd.DataFrame(rows)


# ============================================================
# Main experiment
# ============================================================

results = []
differential_summaries = []
key_sensitivity_summaries = []

for idx, selected_image_path in enumerate(image_files, start=1):
    print(
        f"\n--- Processing Image {idx}/{len(image_files)}: "
        f"{os.path.basename(selected_image_path)} ---"
    )

    img = cv2.imread(
        selected_image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        print("Could not load image. Skipping.")
        continue

    img = cv2.resize(
        img,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    prepared = prepare_keyed_system(
        MASTER_KEY,
        img.shape
    )

    # --------------------------------------------------------
    # Normal encryption: fresh random IV.
    # --------------------------------------------------------
    enc = hybrid_encrypt(
        img,
        MASTER_KEY,
        iv=None,
        prepared=prepared
    )

    dec = hybrid_decrypt(
        enc["ciphertext"],
        img.shape,
        MASTER_KEY,
        enc["iv"],
        prepared=prepared
    )

    encrypted_img = enc["encrypted_image"]
    recovered = dec["recovered_image"]

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(encrypted_img, cmap="gray", vmin=0, vmax=255)
    plt.title("Encrypted")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(recovered, cmap="gray", vmin=0, vmax=255)
    plt.title("Reconstructed")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Core security/statistical metrics
    # --------------------------------------------------------
    corr_original = adjacent_correlations(img)
    corr_encrypted = adjacent_correlations(encrypted_img)

    chi2_stat, chi2_p = perform_chi_square_test(
        encrypted_img
    )

    rec = reconstruction_metrics(
        img,
        recovered
    )

    results.append({
        "Image": os.path.basename(selected_image_path),

        "Entropy_Original": entropy(img),
        "Entropy_Encrypted": entropy(encrypted_img),

        "Corr_H_Original": corr_original["H"],
        "Corr_V_Original": corr_original["V"],
        "Corr_D_Original": corr_original["D"],

        "Corr_H_Encrypted": corr_encrypted["H"],
        "Corr_V_Encrypted": corr_encrypted["V"],
        "Corr_D_Encrypted": corr_encrypted["D"],

        "HistVar_Original": calculate_histogram_variance(img),
        "HistVar_Encrypted": calculate_histogram_variance(encrypted_img),

        "ChiSquare": chi2_stat,
        "ChiSquare_p": chi2_p,

        "MSE": rec["MSE"],
        "PSNR": rec["PSNR"],
        "SSIM": rec["SSIM"],
        "MaxAbsError": rec["MaxAbsError"],
        "PMR_percent": rec["PMR"],
        "ExactReconstruction": bool(np.array_equal(img, recovered)),

        "BSplineScaleUsed": enc["deformation"]["scale"],
        "MinJacobian": enc["deformation"]["min_jacobian"],
        "MaxJacobian": enc["deformation"]["max_jacobian"],

        "Encryption_BSpline_s": enc["time_bspline"],
        "Encryption_Diffusion_s": enc["time_diffusion"],
        "Encryption_AES_s": enc["time_aes"],
        "Encryption_Total_s": enc["time_total"],

        "Decryption_AES_s": dec["time_aes"],
        "Decryption_Diffusion_s": dec["time_diffusion"],
        "Decryption_BSpline_s": dec["time_bspline"],
        "Decryption_Total_s": dec["time_total"],
    })

    # --------------------------------------------------------
    # Differential plaintext sensitivity
    # --------------------------------------------------------
    diff_trials_df, diff_summary = differential_test(
        img,
        MASTER_KEY,
        trials=DIFF_TRIALS,
        alpha=DIFF_ALPHA,
        rng_seed=20260813 + idx
    )

    diff_summary["Image"] = os.path.basename(
        selected_image_path
    )
    differential_summaries.append(diff_summary)

    diff_trials_df.to_csv(
        f"differential_trials_image_{idx}.csv",
        index=False
    )

    # --------------------------------------------------------
    # One-bit master-key sensitivity
    # --------------------------------------------------------
    key_df, key_summary = key_sensitivity_test(
        img,
        MASTER_KEY
    )

    key_summary["Image"] = os.path.basename(
        selected_image_path
    )
    key_sensitivity_summaries.append(key_summary)

    key_df.to_csv(
        f"key_sensitivity_image_{idx}.csv",
        index=False
    )

# ============================================================
# Tables
# ============================================================

results_df = pd.DataFrame(results)
differential_df = pd.DataFrame(differential_summaries)
key_sensitivity_df = pd.DataFrame(key_sensitivity_summaries)

print("\n================ MAIN RESULTS =================\n")
print(results_df.to_string(index=False))

print("\n================ DIFFERENTIAL RESULTS =================\n")
print(differential_df.to_string(index=False))

print("\n================ KEY-SENSITIVITY RESULTS =================\n")
print(key_sensitivity_df.to_string(index=False))

# Save paper-ready raw results.
results_df.to_csv("main_results_v5_nonlinear_diffusion.csv", index=False)
differential_df.to_csv("differential_summary_v5_nonlinear_diffusion.csv", index=False)
key_sensitivity_df.to_csv("key_sensitivity_summary_v5_nonlinear_diffusion.csv", index=False)

# ============================================================
# Overall summaries
# ============================================================

if not results_df.empty:
    print("\n================ OVERALL MAIN SUMMARY =================\n")

    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    summary_main = pd.DataFrame({
        "Mean": results_df[numeric_cols].mean(),
        "SD": results_df[numeric_cols].std(ddof=1),
        "Min": results_df[numeric_cols].min(),
        "Max": results_df[numeric_cols].max(),
    })

    print(summary_main.to_string())
    summary_main.to_csv("main_results_overall_v5_nonlinear_diffusion.csv")


if not differential_df.empty:
    print("\n================ DIFFERENTIAL TEST CRITERIA =================\n")

    th = differential_randomness_thresholds(
        IMAGE_SIZE,
        IMAGE_SIZE,
        DIFF_ALPHA
    )

    print(f"Image size              : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Significance alpha      : {DIFF_ALPHA}")
    print(f"Expected NPCR           : {th['NPCR_expected']:.6f}%")
    print(f"NPCR lower critical     : {th['NPCR_lower']:.6f}%")
    print(f"Expected UACI           : {th['UACI_expected']:.6f}%")
    print(
        f"UACI acceptance interval: "
        f"[{th['UACI_low']:.6f}%, {th['UACI_high']:.6f}%]"
    )


# ============================================================
# Ablation study on first selected image
# ============================================================

if image_files:
    first_img = cv2.imread(
        image_files[0],
        cv2.IMREAD_GRAYSCALE
    )
    first_img = cv2.resize(
        first_img,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    ablation_df = ablation_test(
        first_img,
        MASTER_KEY
    )

    print("\n================ ABLATION STUDY =================\n")
    print(ablation_df.to_string(index=False))

    ablation_df.to_csv(
        "ablation_study_v5_nonlinear_diffusion.csv",
        index=False
    )

print("\nAll v5 nonlinear-diffusion experiment files have been generated.")
print("Nonlinear diffusion inverse self-test: PASSED")
print(
    "IMPORTANT: because the algorithm itself has changed, "
    "the old paper tables must not be reused."
)

import pandas as pd
import glob
import os

all_trials = []

for filename in sorted(glob.glob("differential_trials_image_*.csv")):
    df = pd.read_csv(filename)
    df["SourceFile"] = os.path.basename(filename)
    all_trials.append(df)

all_df = pd.concat(all_trials, ignore_index=True)

catastrophic = all_df[all_df["NPCR"] < 95].copy()

print("Total differential trials:", len(all_df))
print("Trials with NPCR < 95%:", len(catastrophic))

print(
    "Catastrophic-outlier rate:",
    100 * len(catastrophic) / len(all_df),
    "%"
)

print("\nCatastrophic trials:")
print(
    catastrophic[
        [
            "SourceFile",
            "Trial",
            "PixelRow",
            "PixelCol",
            "NPCR",
            "UACI"
        ]
    ].to_string(index=False)
)

print("\nOverall median NPCR:", all_df["NPCR"].median())
print("Overall mean NPCR:", all_df["NPCR"].mean())
print("Overall median UACI:", all_df["UACI"].median())
print("Overall mean UACI:", all_df["UACI"].mean())