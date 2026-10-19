"""Tests for the copy-paste scenes."""

import numpy as np

from door_counter.copy_paste import SCENE_SIZE_PX, CutOut, make_scene, mask_to_line


def test_a_mask_becomes_its_outline() -> None:
    mask = np.zeros((100, 200), dtype=bool)
    mask[20:60, 50:150] = True
    line = mask_to_line(mask=mask, class_index=3)
    assert line is not None
    xy = np.asarray(line.split()[1:], dtype=float).reshape(-1, 2)
    assert line.split()[0] == "3"
    assert np.allclose(xy.min(axis=0), (0.25, 0.2), atol=0.01) and np.allclose(xy.max(axis=0), (0.745, 0.59), atol=0.01)


def test_scenes_have_labels_for_the_pasted_items() -> None:
    background = np.full((SCENE_SIZE_PX[1], SCENE_SIZE_PX[0], 3), 90, dtype=np.uint8)
    can = CutOut(pixels=np.full((60, 40, 3), 200, dtype=np.uint8), mask=np.full((60, 40), 255, dtype=np.uint8),
                 class_index=3)
    rng = np.random.default_rng(0)
    for _ in range(20):
        scene, lines = make_scene(background=background, cut_outs=[[], [], [], [can]], rng=rng)
        assert scene.shape == background.shape
        assert 1 <= len(lines) <= 6 and all(line.startswith("3 ") for line in lines)
        assert (scene > 150).any()  # the bright item is in the picture
