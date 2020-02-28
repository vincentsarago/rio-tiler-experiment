"""Setup cogeo-tiler"""

from setuptools import setup, find_packages

# Runtime requirements.
inst_reqs = ["rio-tiler~=1.3", "rio-color"]

setup(
    name="cog-tiler",
    version="0.0.1",
    python_requires=">=3",
    keywords="COG COGEO Mosaic GIS",
    author=u"Vincent Sarago",
    author_email="vincent@developmentseed.org",
    license="MIT",
    package_data={"app": ["cmaps/*.npy"]},
    packages=find_packages(exclude=["ez_setup", "examples", "tests"]),
    include_package_data=True,
    zip_safe=False,
    install_requires=inst_reqs,
)
