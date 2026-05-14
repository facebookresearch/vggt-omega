# Image Preprocessing

VGGT-Omega uses `load_and_preprocess_images` for simple image loading and
resizing.

```python
from vggt_omega.utils.load_fn import load_and_preprocess_images

images = load_and_preprocess_images(
    image_paths,
    mode="balanced",
    image_resolution=512,
    patch_size=16,
)
```

The returned tensor has shape `[S, 3, H, W]` with values in `[0, 1]`.

## Balanced Mode

`balanced` is the default mode. It keeps the total number of image tokens close
to the requested square resolution while preserving a non-square aspect ratio.
For example, with `image_resolution=512` and `patch_size=16`, it targets about
`32 * 32` patch tokens.

This matches the release inference path used for the example frames.

## Max Size Mode

`max_size` resizes the longest side to `image_resolution` and rounds both
dimensions to multiples of `patch_size`.

```python
images = load_and_preprocess_images(
    image_paths,
    mode="max_size",
    image_resolution=512,
)
```

## Aspect Ratio Crop

Before resizing, both modes center-crop extreme aspect ratios into the range
`[0.5, 2.0]`, where aspect ratio is `height / width`.

This keeps very wide or very tall inputs from producing unusually small token
grids on one axis.

## Mixed Shapes

If a sequence contains images with different resized shapes, the loader pads
them to a common size with white pixels.
