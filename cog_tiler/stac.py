"""STAC reader."""

from typing import Any, Dict, List, Optional, Tuple, Union

import os
from concurrent import futures
import multiprocessing

import numpy

import mercantile

from rasterio.crs import CRS
from rio_tiler.utils import tile_read, tile_exists
from rio_tiler.errors import TileOutsideBounds


from cog_tiler.base import reader as baseReader

WEB_MERCATOR_CRS = CRS.from_epsg(3857)
WGS84_CRS = CRS.from_epsg(4326)
MAX_THREADS = int(os.environ.get("MAX_THREADS", multiprocessing.cpu_count() * 5))


class Reader(baseReader):
    """
    STAC Reader.

    Examples
    --------

    Attributes
    ----------
        src_path : str
            Cloud Optimized GeoTIFF path.

    Methods
    -------
        tile_exists(0, 0, 0)
            Check if a mercator tile intersects with the COG bounds.
        tile(0, 0, 0, indexes=(1,2,3), tilesize=512, resampling_methods="nearest")
            Read tile from a COG.
        expression(0, 0, 0, "b1/b2", tilesize=512, resampling_methods="nearest")
            Read tile and apply simple band math.
        raw(0, 0, 0, indexes=(1,2,3), tilesize=512, resampling_methods="nearest")
            Read tile from a COG and return as GeoTIFF.
        metadata(pmin=5, pmax=95)
        clip((0,10,0,10), indexes=(1,2,3,))
        point((10, 10), indexes=1)
    """

    def __init__(self, item: Dict):
        self.item = item
        self._bounds = self.item["bbox"]
        self._asset_names = list(self.item["assets"].keys())

    def __enter__(self):
        return self

    def _read_multiple(
        self,
        assets: List[str],
        x: int,
        y: int,
        z: int,
        tilesize: int = 256,
        **kwargs: Any,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Read multiple assets and merge the arrays."""
        tile_bounds = mercantile.xy_bounds(mercantile.Tile(x=x, y=y, z=z))

        def worker(asset: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
            return tile_read(asset, tile_bounds, tilesize=tilesize, **kwargs)

        with futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            data, masks = zip(*list(executor.map(worker, assets)))
            data = numpy.concatenate(data)
            mask = numpy.all(masks, axis=0).astype(numpy.uint8) * 255

        return data, mask

    def tile_exists(self, x: int, y: int, z: int) -> bool:
        """
        Check if the tile exist.

        Attributes
        ----------
            x : int
                X tile map index.
            y : int
                y tile map index
            z : int
                Z tile map index

        Returns
        -------
            bool
        """
        return tile_exists(self._bounds, z, x, y)

    def tile(
        self,
        x: int,
        y: int,
        z: int,
        tilesize: int = 256,
        assets: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Read Tile from a COG.

        Attributes
        ----------
            x : int
                X tile map index.
            y : int
                y tile map index
            z : int
                Z tile map index
            assets : list or tuple or int, optional
                Band assets.
            tilesize : int, optional
                Output tile size. Default is 256.
            kwargs : Any, optional
                Additional options to forward to rio_tiler.utils._tile_read

        Returns
        -------
            tile : numpy.ndarray
                Tile data.
            mask : numpy.ndarray
                Mask data.

        """
        if isinstance(assets, str):
            assets = [assets]
        elif isinstance(assets, tuple):
            assets = list(assets)

        if not self.tile_exists(x, y, z):
            raise TileOutsideBounds(f"Tile {z}/{x}/{y} is outside image bounds")

        assets = assets if assets else list(self.item["assets"].keys())
        assets = [self.item["assets"][asset]["href"] for asset in assets]
        return self._read_multiple(assets, x, y, z, tilesize=tilesize, **kwargs)

    def expression(
        self, x: int, y: int, z: int, expr: str, tilesize: int = 256, **kwargs: Any
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        raise Exception("Not implemented")

    def clip(
        self,
        bbox: Tuple[float, ...],
        max_size: int = 1024,
        box_crs: CRS = WGS84_CRS,
        dst_crs: CRS = WEB_MERCATOR_CRS,
        assets: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        raise Exception("Not implemented")

    def point(
        self,
        coordinates: Tuple[float, float],
        coord_crs: CRS = WGS84_CRS,
        assets: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List:
        raise Exception("Not implemented")

    def metadata(
        self,
        pmin: float = 2.0,
        pmax: float = 98.0,
        histogram_bins: Union[int, List, Tuple[int, ...]] = 20,
        **kwargs: Any,
    ) -> Dict:
        raise Exception("Not implemented")
