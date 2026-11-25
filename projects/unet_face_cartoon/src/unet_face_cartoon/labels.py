"""CelebAMask-HQ 官方 face_parsing 的编号，不可与 BiSeNet 编号混用。"""

from enum import IntEnum


class FaceLabel(IntEnum):
    BACKGROUND = 0
    SKIN = 1
    NOSE = 2
    EYEGLASSES = 3
    LEFT_EYE = 4
    RIGHT_EYE = 5
    LEFT_BROW = 6
    RIGHT_BROW = 7
    LEFT_EAR = 8
    RIGHT_EAR = 9
    MOUTH = 10
    UPPER_LIP = 11
    LOWER_LIP = 12
    HAIR = 13
    HAT = 14
    EARRING = 15
    NECKLACE = 16
    NECK = 17
    CLOTH = 18


SKIN_EDIT_LABELS = (FaceLabel.SKIN,)
DETAIL_LABELS = (
    FaceLabel.NOSE, FaceLabel.EYEGLASSES,
    FaceLabel.LEFT_EYE, FaceLabel.RIGHT_EYE,
    FaceLabel.LEFT_BROW, FaceLabel.RIGHT_BROW,
    FaceLabel.MOUTH, FaceLabel.UPPER_LIP, FaceLabel.LOWER_LIP,
)
# 候选头像口径，不宣称等价于原模型的二值头像分割器。
HEAD_LABELS = tuple(label for label in FaceLabel if 1 <= label.value <= 15)
