"""rio-tiler-color."""

from typing import Any, Dict, List, Optional, Tuple, Union

import re

import numpy
import numexpr

import mercantile

import rasterio
from rasterio.io import MemoryFile
from rasterio.crs import CRS
from rasterio.warp import transform, transform_bounds
from rasterio.enums import ColorInterp, MaskFlags

from rio_tiler.utils import raster_get_stats, tile_exists, _tile_read

from cog_tiler.utils import geotiff_options

WEB_MERCATOR_CRS = CRS.from_epsg(3857)
WGS84_CRS = CRS.from_epsg(4326)


# Should be in rio-tiler
def has_mask_band(src_dst):
    """Check for mask band in source."""
    if any([MaskFlags.per_dataset in flags for flags in src_dst.mask_flag_enums]):
        return True
    return False


class cogTiler(object):
    """
    Cloud Optimized GeoTIFF Tiler.


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

    def __init__(self, src_path: str):
        self.src_path: str = src_path
        self.colormap: Dict[Any, Any]
        self.bounds: Tuple[float, ...]

    def __enter__(self):
        self.src_dst = rasterio.open(self.src_path)
        if self.src_dst.colorinterp[0] is ColorInterp.palette:
            try:
                self.colormap = self.src_dst.colormap(1)
            except ValueError:
                pass

        self.bounds = transform_bounds(
            self.src_dst.crs, WGS84_CRS, *self.src_dst.bounds, densify_pts=21
        )

        return self

    def __exit__(self, *args):
        self.src_dst.close()

    def tile_exists(self, x: int, y: int, z: int) -> bool:
        return tile_exists(self.bounds, z, x, y)

    def tile(
        self,
        x: int,
        y: int,
        z: int,
        indexes: Optional[Union[Tuple, List, int]] = None,
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
            indexes : list or tuple or int, optional
                Band indexes.
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
        tile_bounds = mercantile.xy_bounds(mercantile.Tile(x=x, y=y, z=z))
        return _tile_read(
            self.src_dst, tile_bounds, tilesize, indexes=indexes, **kwargs
        )

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
        bands_names = tuple(set(re.findall(r"b(?P<bands>[0-9A]{1,2})", expr)))
        expr_bands = ["b{}".format(b) for b in bands_names]

        indexes = tuple(map(int, bands_names))
        tile, mask = self.tile(x, y, z, indexes=indexes, tilesize=tilesize, **kwargs)

        tile = dict(zip(expr_bands, tile))
        rgb = expr.split(",")
        return (
            numpy.array(
                [
                    numpy.nan_to_num(numexpr.evaluate(bloc.strip(), local_dict=tile))
                    for bloc in rgb
                ]
            ),
            mask,
        )

    def raw(
        self,
        x: int,
        y: int,
        z: int,
        indexes: Optional[Union[Tuple, List, int]] = None,
        tilesize: int = 256,
        **kwargs: Any,
    ) -> bytes:
        """
        Read Tile from a COG and render as a GeoTIFF file.

        Attributes
        ----------
            x : int
                X tile map index.
            y : int
                y tile map index
            z : int
                Z tile map index
            indexes : list or tuple or int, optional
                Band indexes.
            tilesize : int, optional
                Output tile size. Default is 256
            kwargs : Any, optional
                Additional options to forward to rio_tiler.utils._tile_read

        Returns
        -------
            bytes : GeoTIFF binary content

        """
        if isinstance(indexes, int):
            indexes = (indexes,)

        indexes = indexes if indexes else self.src_dst.indexes

        tile, alpha = self.tile(x, y, z, indexes=indexes, tilesize=tilesize, **kwargs)

        count, _, _ = tile.shape
        output_profile = dict(
            driver="GTiff",
            dtype=tile.dtype,
            count=count,
            nodata=self.src_dst.nodata,
            height=tilesize,
            width=tilesize,
        )
        options = geotiff_options(x, y, z, tilesize)
        output_profile.update(options)

        with MemoryFile() as memfile:
            with memfile.open(**output_profile) as dst:
                dst.write(tile, indexes=list(range(1, count + 1)))
                if has_mask_band(self.src_dst):
                    dst.write_mask(alpha.astype("uint8"))

                dst.colorinterp = [self.src_dst.colorinterp[b - 1] for b in indexes]
                if self.colormap:
                    dst.write_colormap(1, self.colormap)

                for i, b in enumerate(indexes):
                    dst.set_band_description(i + 1, self.src_dst.descriptions[b - 1])
                    dst.update_tags(i + 1, **self.src_dst.tags(b))

                dst.update_tags(**self.src_dst.tags())
                dst._set_all_scales([self.src_dst.scales[b - 1] for b in indexes])
                dst._set_all_offsets([self.src_dst.offsets[b - 1] for b in indexes])

            return memfile.read()

    # https://github.com/cogeotiff/rio-tiler/issues/152
    def clip(
        self,
        bbox: Tuple[float, ...],
        indexes: Optional[Union[Tuple, List, int]] = None,
        max_size: int = 1024,
        box_crs: CRS = WGS84_CRS,
        dst_crs: CRS = WEB_MERCATOR_CRS,
        **kwargs: Any,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Read Tile from a COG and render as a GeoTIFF file.

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
        raise Exception("Not implemented")

    def point(
        self,
        coordinates: Tuple[float, float],
        indexes: Optional[Union[Tuple, List, int]] = None,
        coord_crs: CRS = WGS84_CRS,
    ) -> List:
        """
        Read Tile from a COG and render as a GeoTIFF file.

        Attributes
        ----------
            coordinates : tuple
                (X, Y) coordinates.
            indexes : list or tuple or int, optional
                Band indexes.
            coord_crs : rasterio.crs.CRS, optional
                (X, Y) coordinate system. Default is WGS84/EPSG:4326.

        Returns
        -------
            list : List

        """
        indexes = indexes if indexes else self.src_dst.indexes

        lon, lat = transform(
            coord_crs, self.src_dst.crs, [coordinates[0]], [coordinates[1]]
        )
        return list(self.src_dst.sample([(lon[0], lat[0])], indexes=indexes))[
            0
        ].tolist()

    def metadata(
        self,
        pmin: float = 2.0,
        pmax: float = 98.0,
        histogram_bins: Union[int, List, Tuple[int, ...]] = 20,
        **kwargs: Any,
    ):
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
        meta = dict(url=self.src_path, dtype=self.src_dst.dtypes[0])
        if self.colormap:
            meta.update(dict(colormap=self.colormap))
            histogram_bins = [
                k for k, v in self.colormap.items() if v != (0, 0, 0, 255)
            ]

        meta.update(
            raster_get_stats(
                self.src_dst,
                percentiles=(pmin, pmax),
                histogram_bins=histogram_bins,
                **kwargs,
            )
        )

        return meta
