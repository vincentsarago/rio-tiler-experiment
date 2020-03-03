"""Abastract Class."""

from typing import Any, Dict, List, Tuple, Union

import abc

import numpy
from rasterio.crs import CRS

WEB_MERCATOR_CRS = CRS.from_epsg(3857)
WGS84_CRS = CRS.from_epsg(4326)


class reader(abc.ABC):
    """Abstract base class."""

    @abc.abstractmethod
    def tile(
        self,
        x: int,
        y: int,
        z: int,
        tilesize: int = 256,
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

    @abc.abstractmethod
    def expression(
        self, x: int, y: int, z: int, expr: str, tilesize: int = 256, **kwargs: Any
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Read Tile from a COG and apply simple band math.

        Attributes
        ----------
            x : int
                X tile map index.
            y : int
                y tile map index
            z : int
                Z tile map index
            expr : str
                Band math expression.
            tilesize : int, optional
                Output tile size. Default is 256
            kwargs : Any, optional
                Additional options to forward to rio_tiler.utils._tile_read

        Returns
        -------
            tile : numpy.ndarray
                Tile data.
            mask : numpy.ndarray
                Mask data.

        """

    @abc.abstractmethod
    def clip(
        self,
        bbox: Tuple[float, ...],
        max_size: int = 1024,
        box_crs: CRS = WGS84_CRS,
        dst_crs: CRS = WEB_MERCATOR_CRS,
        **kwargs: Any,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Read part of a COG.

        Attributes
        ----------
            bbox : tuple
                (xmin, ymin, xmax, ymax) bounding box.
            indexes : list or tuple or int, optional
                Band indexes.
            max_size : int, optional
                Output array maximum size. Default is 1024.
            box_crs : rasterio.crs.CRS, optional
                Bounding box coordinate system. Default is WGS84/EPSG:4326.
            dst_crs : rasterio.crs.CRS, optional
                Output coordinate system. Default is WebMercator/EPSG:3857.
            kwargs : Any, optional
                Additional options to forward to rio_tiler.utils._tile_read

        Returns
        -------
            tile : numpy.ndarray
                Data.
            mask : numpy.ndarray
                Mask data.

        """

    @abc.abstractmethod
    def point(
        self,
        coordinates: Tuple[float, float],
        coord_crs: CRS = WGS84_CRS,
        **kwargs: Any,
    ) -> List:
        """
        Read point value from a COG.

        Attributes
        ----------
            coordinates : tuple
                (X, Y) coordinates.
            coord_crs : rasterio.crs.CRS, optional
                (X, Y) coordinate system. Default is WGS84/EPSG:4326.
            kwargs : Any, optional
                Additional options to forward to src_dst.sample()

        Returns
        -------
            list : List

        """

    @abc.abstractmethod
    def metadata(
        self,
        pmin: float = 2.0,
        pmax: float = 98.0,
        histogram_bins: Union[int, List, Tuple[int, ...]] = 20,
        **kwargs: Any,
    ) -> Dict:
        """
        Compute array statistics.

        see https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
        for detailed options

        Attributes
        ----------
            pmin : float, optional
                Min percentile. Default is 2.0.
            pmax : float, optional
                Max percentile. Default is 98.0.
            histogram_bins : int or sequence of scalars or str, optional
                Histogram bins. Default is 10.
            kwargs : Any, optional
                Additional options to forward to rio_tiler.utils.raster_get_stats

        Returns
        -------
            metadata : dict

        """