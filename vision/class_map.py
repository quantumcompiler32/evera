"""Versioned YOLO segmentation class orders; existing IDs are append-only."""

LEGACY_CLASSES = ("raspberry_pi", "arduino", "pinecil", "wire")
TARGET_CLASSES = LEGACY_CLASSES + ("tape_measure", "multimeter")
SUPPORTED_CLASS_ORDERS = (LEGACY_CLASSES, TARGET_CLASSES)
