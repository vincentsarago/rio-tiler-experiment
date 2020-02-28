"""Commons."""

extensions = dict(JPEG="jpg", PNG="png", GTiff="tif", WEBP="webp")

drivers = dict(jpg="JPEG", png="PNG", pngraw="PNG", tif="GTiff", webp="WEBP")

mimetype = dict(
    png="image/png",
    pngraw="image/png",
    npy="application/x-binary",
    tif="image/tiff",
    jpg="image/jpg",
    webp="image/webp",
)
