"""cogeo_tiler.handler: handle request for cogeo-tiler."""

from typing import Any, Dict, Optional, Tuple, Union

import io
import os
import re
import json
from urllib.parse import urlencode

import numpy
from boto3.session import Session as boto3_session

import rasterio
from rasterio import warp
from rasterio.session import AWSSession

from rio_tiler.mercator import get_zooms
from rio_tiler.profiles import img_profiles

from cog_tiler import cmap
from cog_tiler import utils
from cog_tiler.cogeo import CogeoReader

from .common import drivers, mimetype

from lambda_proxy.proxy import API

app = API(name="app")
aws_session = AWSSession(session=boto3_session())


class TilerError(Exception):
    """Base exception class."""


@app.route(
    "/metadata",
    methods=["GET"],
    cors=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    tag=["metadata"],
)
def metadata_handler(
    url: str,
    pmin: float = 2.0,
    pmax: float = 98.0,
    nodata: float = None,
    indexes: Optional[Tuple[int]] = None,
    overview_level: Optional[int] = None,
    max_size: int = 1024,
    histogram_bins: int = 20,
    histogram_range: Optional[int] = None,
    resampling_method: str = "nearest",
) -> Tuple[str, str, str]:
    """Handle /metadata requests."""
    pmin = float(pmin) if isinstance(pmin, str) else pmin
    pmax = float(pmax) if isinstance(pmax, str) else pmax
    if nodata is not None and isinstance(nodata, str):
        nodata = numpy.nan if nodata == "nan" else float(nodata)
    if indexes is not None and isinstance(indexes, str):
        indexes = tuple(int(s) for s in re.findall(r"\d+", indexes))

    if overview_level is not None and isinstance(overview_level, str):
        overview_level = int(overview_level)

    max_size = int(max_size) if isinstance(max_size, str) else max_size
    histogram_bins = (
        int(histogram_bins) if isinstance(histogram_bins, str) else histogram_bins
    )

    if histogram_range is not None and isinstance(histogram_range, str):
        histogram_range = tuple(map(float, histogram_range.split(",")))

    with rasterio.Env(aws_session):
        with CogeoReader(url) as cog:
            meta = cog.metadata(
                pmin,
                pmax,
                nodata=nodata,
                indexes=indexes,
                resampling_method=resampling_method,
                overview_level=overview_level,
                histogram_bins=histogram_bins,
                histogram_range=histogram_range,
            )

        return ("OK", "application/json", json.dumps(meta))


@app.route(
    "/bounds",
    methods=["GET"],
    cors=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    tag=["tiles"],
)
def bounds_handler(url: str) -> Tuple[str, str, str]:
    """Handle /bounds requests."""
    with rasterio.Env(aws_session):
        with CogeoReader(url) as cog:
            bounds = cog.bounds

    meta = dict(bounds=bounds, url=url)
    return ("OK", "application/json", json.dumps(meta))


@app.route(
    "/tilejson.json",
    methods=["GET"],
    cors=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    tag=["tiles"],
)
def tilejson_handler(
    url: str, tile_scale: int = 1, tile_format: str = None, **kwargs: Any
) -> Tuple[str, str, str]:
    """Handle /tilejson.json requests."""
    if tile_scale is not None and isinstance(tile_scale, str):
        tile_scale = int(tile_scale)

    kwargs.update(dict(url=url))
    qs = urlencode(list(kwargs.items()))
    if tile_format:
        tile_url = f"{app.host}/{{z}}/{{x}}/{{y}}@{tile_scale}x.{tile_format}?{qs}"
    else:
        tile_url = f"{app.host}/{{z}}/{{x}}/{{y}}@{tile_scale}x?{qs}"

    with rasterio.Env(aws_session):
        with rasterio.open(url) as src_dst:
            bounds = list(
                warp.transform_bounds(
                    src_dst.crs, "epsg:4326", *src_dst.bounds, densify_pts=21
                )
            )
            minzoom, maxzoom = get_zooms(src_dst)
            center = [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2, minzoom]

    meta = dict(
        bounds=bounds,
        center=center,
        minzoom=minzoom,
        maxzoom=maxzoom,
        name=os.path.basename(url),
        tilejson="2.1.0",
        tiles=[tile_url],
    )
    return ("OK", "application/json", json.dumps(meta))


@app.route(
    "/<int:z>/<int:x>/<int:y>.<ext>",
    methods=["GET"],
    cors=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    tag=["tiles"],
)
@app.route(
    "/<int:z>/<int:x>/<int:y>",
    methods=["GET"],
    cors=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    tag=["tiles"],
)
@app.route(
    "/<int:z>/<int:x>/<int:y>@<int:scale>x.<ext>",
    methods=["GET"],
    cors=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    tag=["tiles"],
)
@app.route(
    "/<int:z>/<int:x>/<int:y>@<int:scale>x",
    methods=["GET"],
    cors=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    tag=["tiles"],
)
def tile_handler(
    z: int,
    x: int,
    y: int,
    scale: int = 1,
    ext: str = None,
    url: str = None,
    indexes: Optional[Tuple[int]] = None,
    expression: str = None,
    nodata: Union[str, int, float] = None,
    rescale: str = None,
    color_formula: str = None,
    color_map: str = None,
    alpha_classes: Tuple[int] = None,  # Experiment
    ignore_classes: Tuple[int] = None,  # Experiment
    resampling_method: str = "nearest",
) -> Tuple[str, str, bytes]:
    """Handle /tiles requests."""
    if indexes and expression:
        raise TilerError("Cannot pass indexes and expression")

    if not url:
        raise TilerError("Missing 'url' parameter")

    if isinstance(indexes, str):
        indexes = tuple(int(s) for s in re.findall(r"\d+", indexes))

    if isinstance(alpha_classes, str):
        alpha_classes = tuple(int(s) for s in alpha_classes.split(","))

    if nodata is not None:
        nodata = numpy.nan if nodata == "nan" else float(nodata)

    if scale is not None and isinstance(scale, str):
        scale = int(scale)

    tilesize: int = scale * 256

    tile_params: Dict = dict(
        tilesize=tilesize, nodata=nodata, resampling_method=resampling_method
    )

    colormap = cmap.get_colormap(color_map) if color_map else None

    with rasterio.Env(aws_session):
        with CogeoReader(url) as cog:
            if expression is not None:
                tile, mask = cog.expression(x, y, z, expression, **tile_params)
            else:
                tile, mask = cog.tile(x, y, z, indexes=indexes, **tile_params)

                if cog.colormap and not colormap:
                    colormap = cog.colormap
                    if alpha_classes:
                        cmap._update_alpha(colormap, alpha_classes)
                    elif ignore_classes:
                        cmap._remove_value(colormap, ignore_classes)

    if not ext:
        ext = "jpg" if mask.all() else "png"

    tile = utils.postprocess(tile, mask, rescale=rescale, color_formula=color_formula)
    if ext == "npy":
        sio = io.BytesIO()
        numpy.save(sio, (tile, mask))
        sio.seek(0)
        content = sio.getvalue()
    else:
        driver = drivers[ext]
        options = (
            img_profiles.get(driver.lower(), {})
            if not ext == "tif"
            else utils.geotiff_options(x, y, z, tilesize)
        )
        content = utils.render(
            tile, mask, img_format=driver, colormap=colormap, **options
        )

    return ("OK", mimetype[ext], content)


@app.route(
    "/raw/<int:z>/<int:x>/<int:y>.tif",
    methods=["GET"],
    cors=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    tag=["raw"],
)
@app.route(
    "/raw/<int:z>/<int:x>/<int:y>@<int:scale>x.tif",
    methods=["GET"],
    cors=True,
    payload_compression_method="gzip",
    binary_b64encode=True,
    tag=["raw"],
)
def raw_tile(
    z: int,
    x: int,
    y: int,
    scale: int = 1,
    url: str = None,
    indexes: Optional[Tuple[int]] = None,
    nodata: Union[str, int, float] = None,
    resampling_method: str = "nearest",
) -> Tuple[str, str, bytes]:
    """Handle /tiles requests."""
    if not url:
        raise TilerError("Missing 'url' parameter")

    if isinstance(indexes, str):
        indexes = tuple(int(s) for s in re.findall(r"\d+", indexes))

    if nodata is not None:
        nodata = numpy.nan if nodata == "nan" else float(nodata)

    if scale is not None and isinstance(scale, str):
        scale = int(scale)

    tilesize: int = scale * 256

    tile_params: Dict = dict(
        tilesize=tilesize, nodata=nodata, resampling_method=resampling_method
    )

    with rasterio.Env(aws_session):
        with cogeoReader(url) as cog:
            content = cog.raw(x, y, z, indexes=indexes, **tile_params)

    return ("OK", "image/tiff", content)
