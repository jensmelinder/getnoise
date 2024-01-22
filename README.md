# getnoise
Script to estimate depth of JWST images in different ways.

Estimates standard deviation of the sky in JWST images (tested with MIRIM and NIRCAM, should work with NIRISS as well) in various ways and compares them.

The baseline noise estimate is calculated by using a number of circular apertures in sky regions in the image.

## Requirements:
Apart from standard astronomical python packages (astropy, numpy, matplotlib, scipy) also ```sep``` is needed.

## Example usage:
To run, copy the script (and aperture correction files) to the working directory. Start python/ipython/jupyter.
```
from getnoise import get_apernoise
xs = 1100
xe = 2290
ys = 200
ye = 2025
get_apernoise(<JWST fits file name>, <naper>, <pixel scale>, bgdr=5,
              aprad=0.15, apdiam_lim=0.45, detlim=1.0, minarea = 10,
              errstats=True, plots=True, outroot='None',
              xs=xs, xe=xe, ys=ys, ye=ye,
              apcorfname = 'jwst_miri_apcorr_0010.fits', detector = 'MIRIM',
              pixfrac=1.0, bunit = 'MJy/sr', filter='F560W')
```

## Details:

```(xs, xe, ys, ye)``` allows you to select part of the image, omit these if you want the full image.
For details on what the parameters mean, check the code, it should be straight forward to understand what they do.
The current aperture correction calibration files are included and needs to be present in the directory where you run the script.



