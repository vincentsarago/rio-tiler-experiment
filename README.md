# rio-tiler-experiment


```python
from cog_tiler.tiler import cogTiler

...

tilesize: int = scale * 256

tile_params: Dict = dict(
    tilesize=tilesize, nodata=nodata, resampling_method=resampling_method
)

with cogTiler(url) as cog:
    if not cog.tile_exists(x, y, z):
        raise TileOutsideBounds(
            f"Tile {z}/{x}/{y} is outside image bounds"
        )

    if expression is not None:
        tile, mask = cog.expression(x, y, z, expression, **tile_params)
    
    else:
        tile, mask = cog.tile(x, y, z, indexes=indexes, **tile_params)

```