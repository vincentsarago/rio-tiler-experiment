# rio-tiler-experiment

### Simple COG

```python
from cog_tiler.cogeo import Reader as COGReader

...

tilesize: int = scale * 256

tile_params: Dict = dict(
    tilesize=tilesize, nodata=nodata, resampling_method=resampling_method
)

with COGReader(url) as cog:
    if expression is not None:
        tile, mask = cog.expression(x, y, z, expression, **tile_params)
    else:
        tile, mask = cog.tile(x, y, z, indexes=indexes, **tile_params)

# Or with an open dataset
with rasterio.open(url) as src_dst:
    with COGReader(None, dataset=src_dst) as cog:
        if expression is not None:
            tile, mask = cog.expression(x, y, z, expression, **tile_params)
        else:
            tile, mask = cog.tile(x, y, z, indexes=indexes, **tile_params)
```

### STAC

```python
from cog_tiler.stac import Reader as STACReader

...
# Get STAC ITEM

tilesize: int = scale * 256

assets = ["red", "blue", "green"]
with STACReader(my_stac_item) as stac:
    tile, mask = stac.tile(x, y, z, assets=assets, tilesize=tilesize)

```