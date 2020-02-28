"""cog_tiler.utils."""

from typing import Any, Dict, Optional

import numpy

import mercantile

from rasterio.io import MemoryFile
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from rio_color.operations import parse_operations
from rio_color.utils import scale_dtype, to_math_type
from rio_tiler.utils import linear_rescale, _chunks

from cog_tiler.cmap import apply_cmap


def geotiff_options(
    tile_x, tile_y, tile_z, tilesize: int = 256, dst_crs: CRS = CRS.from_epsg(3857)
) -> Dict:
    """GeoTIFF options."""
    bounds = mercantile.xy_bounds(mercantile.Tile(x=tile_x, y=tile_y, z=tile_z))
    dst_transform = from_bounds(*bounds, tilesize, tilesize)
    return dict(crs=dst_crs, transform=dst_transform)


def postprocess(
    tile: numpy.ndarray,
    mask: numpy.ndarray,
    rescale: str = None,
    color_formula: str = None,
) -> numpy.ndarray:
    """Post-process tile data."""
    if rescale:
        rescale_arr = list(map(float, rescale.split(",")))
        rescale_arr = list(_chunks(rescale_arr, 2))
        if len(rescale_arr) != tile.shape[0]:
            rescale_arr = ((rescale_arr[0]),) * tile.shape[0]

        for bdx in range(tile.shape[0]):
            tile[bdx] = numpy.where(
                mask,
                linear_rescale(
                    tile[bdx], in_range=rescale_arr[bdx], out_range=[0, 255]
                ),
                0,
            )
        tile = tile.astype(numpy.uint8)

    if color_formula:
        # make sure one last time we don't have
        # negative value before applying color formula
        tile[tile < 0] = 0
        for ops in parse_operations(color_formula):
            tile = scale_dtype(ops(to_math_type(tile)), numpy.uint8)

    return tile


def render(
    tile: numpy.ndarray,
    mask: Optional[numpy.ndarray] = None,
    img_format: str = "PNG",
    colormap: Optional[Dict] = None,
    **creation_options: Any,
) -> bytes:
    """
    Translate numpy ndarray to image buffer using GDAL.

    Usage
    -----
        tile, mask = rio_tiler.utils.tile_read(......)
        with open('test.jpg', 'wb') as f:
            f.write(array_to_image(tile, mask, img_format="jpeg"))

    Attributes
    ----------
        tile : numpy ndarray
            Image array to encode.
        mask: numpy ndarray, optional
            Mask array
        img_format: str, optional
            Image format to return (default: 'png').
            List of supported format by GDAL: https://www.gdal.org/formats_list.html
        colormap: dict, optional
            GDAL RGBA Color Table dictionary.
        creation_options: dict, optional
            Image driver creation options to pass to GDAL

    Returns
    -------
        bytes: BytesIO
            Reurn image body.

    """
    if len(tile.shape) < 3:
        tile = numpy.expand_dims(tile, axis=0)

    if colormap:
        tile, alpha = apply_cmap(tile, colormap)
        if mask is not None:
            mask = (
                mask * alpha * 255
            )  # This is a special case when we want to mask some valid data

    # WEBP doesn't support 1band dataset so we must hack to create a RGB dataset
    if img_format == "WEBP" and tile.shape[0] == 1:
        tile = numpy.repeat(tile, 3, axis=0)
    elif img_format == "JPEG":
        mask = None

    count, height, width = tile.shape

    output_profile = dict(
        driver=img_format,
        dtype=tile.dtype,
        count=count + 1 if mask is not None else count,
        height=height,
        width=width,
    )
    output_profile.update(creation_options)

    with MemoryFile() as memfile:
        with memfile.open(**output_profile) as dst:
            dst.write(tile, indexes=list(range(1, count + 1)))
            # Use Mask as an alpha band
            if mask is not None:
                dst.write(mask.astype(tile.dtype), indexes=count + 1)

        return memfile.read()
