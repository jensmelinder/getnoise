from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table
from astropy.coordinates import match_coordinates_sky, SkyCoord
import sep
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
import random
import matplotlib
from scipy.interpolate import make_interp_spline
from copy import copy
import sys
import warnings

def avsigclip(data,niter,sig):
    """
    Computes the average in a numpy array after sigma clipping.
    """
    mean = np.nanmean(data)
    stdev= np.nanstd(data)
    for ii in range(niter):
        data = data[(data>(mean - sig*stdev))&(data<(mean + sig*stdev))]
        mean = np.nanmean(data)
        stdev= np.nanstd(data)
    return mean,stdev

def get_apcor(rad, apcorfname, pscale, filter, plot=True):
    '''Get aperture correction at radius in arcsec.'''
    apcordata = fits.getdata(apcorfname)
    if ('miri' in apcorfname):
        mm = (apcordata['filter'] == filter) & (apcordata['subarray'] =='FULL')
    elif ('nircam' in apcorfname):
        mm = (apcordata['filter'] == filter) & (apcordata['pupil'] =='CLEAR')
    elif ('niriss' in apcorfname):
        mm = (apcordata['pupil'] == filter)
    else:
        warnings.warn(f"WARNING: Only MIRI/NIRCAM/NIRISS implemented. Using NIRCAM setup (which may fail).")
        mm = (apcordata['filter'] == filter)

    rad_apcor = np.array([rr[3] for rr in apcordata[mm]])*pscale # arcsec
    apcor = np.array([rr[4] for rr in apcordata[mm]])

    spl_apcor = make_interp_spline(rad_apcor, apcor, k=2)

    # get measurements for aperture
    apcor_rmeas = spl_apcor(rad)

    if plot:
        fig,ax = plt.subplots(figsize=(8,5))
        ax.plot(rad_apcor, apcor, ls='None', marker='x', mew=2)
        plotrad = np.arange(rad_apcor[0],rad_apcor[-1],pscale/10)
        ax.plot(plotrad, spl_apcor(plotrad), lw=2, color='red')
        ax.axhline(apcor_rmeas,color='k',lw=2,ls=':')
        
    return apcor_rmeas.flatten()[0]

# function to calculate the stats
def get_apernoise(imroot, naper, pscale, bgdr=5, aprad=0.15, apdiam_lim=0.45, detlim=1.0, minarea=10, errstats=True, plots=True, 
                  outroot='None', xs=-1, xe=-1, ys=-1, ye=-1, apcorfname = 'jwst_miri_apcorr_0010.fits', 
                  detector='MIRIMAGE', pixfrac=1.0, bunit = 'MJy/sr', filter='F560W'):
    '''Function that calculates aperture statistics on a JWST image. Uses HST style filenames (sci/wht).'''
    # read data
    im=fits.getdata(imroot+'_sci.fits', 0).byteswap().newbyteorder()
    head = fits.getheader(imroot+'_sci.fits', 0)
    wht=fits.getdata(imroot+'_wht.fits', 0).byteswap().newbyteorder()
    ima = np.ascontiguousarray(im)
    wht = np.ascontiguousarray(wht)
    mask = np.where(ima==0, 1., 0.)
    warnings.filterwarnings('ignore')
    err = np.sqrt(1./np.where(mask==1,1,wht))
    err = np.where(mask==1,0,err)
    warnings.filterwarnings('default')
    if (xs!=-1):
        # cut out region with high depth
        ima = np.ascontiguousarray(ima[ys:ye,xs:xe])
        err = np.ascontiguousarray(err[ys:ye,xs:xe])
        mask = np.ascontiguousarray(mask[ys:ye,xs:xe])

    if (outroot=='None'):
        outroot = imroot

    # check unit, only 10nanoJy, muJy, and MJy/sr supported, internally converted to muJy
    if (bunit == 'MJy/sr'):
        try:
            pixarsr = head['PIXAR_SR']
        except:
            pixarsr = head['PIXARSR']
        calfactor = pixarsr * 1e12 # to get uJy from MJy/sr
    elif (bunit == 'muJy'):
        calfactor = 1.
    elif (bunit == '10nanoJy'):
        calfactor = 1e-2
    else:
        warnings.filterwarnings("ignore")
        raise SystemExit('Unsupported bunit. Must be one of ["MJy/sr", "muJy", or "10nanoJy"]. Aborting.')

    # set native pixel scale depending on detector
    if (detector == 'MIRIMAGE'):
        natscale = 0.11
    elif (detector == 'NIRCAM-SW'):
        natscale = 0.031
    elif (detector == 'NIRCAM-LW'):
        natscale = 0.063
    elif (detector == 'NIRISS'):
        natscale = 0.0655
    else:
        warnings.filterwarnings("ignore")
        raise SystemExit('Only JWST:NIRCAM/MIRI/NIRISS images are supported. Aborting.')

    # make source mask
    bkg = sep.Background(ima, mask=mask, bw=128, bh=128, fw=3, fh=3)
    sub = ima - bkg
    sub = np.where(mask==1., 0, sub)
    objects,segmap = sep.extract(sub, detlim, err=err, mask=mask, minarea=minarea, filter_type='matched', deblend_nthresh=64, 
                          deblend_cont=0.001, clean=True, clean_param=1.0, segmentation_map=True)
    sourcemask = np.where(segmap==0, 0, 1.)
    # get array of aperture "fluxes" from all pixels
    aprad = aprad/pscale # pixels
    aprad_seg = aprad+bgdr+1 
    xrange = np.arange(aprad_seg+1,sub.shape[0]-aprad_seg-1,2*(aprad_seg+1))
    yrange = np.arange(aprad_seg+1,sub.shape[1]-aprad_seg-1,2*(aprad_seg+1))

    selected_pos = []
    for xi in xrange:
        for yi in yrange:
            # shift x and y here, apply subpixel shift
            dx = random.random()-0.5
            dy = random.random()-0.5
            aper = CircularAperture((yi+dy,xi+dx), r=aprad_seg)
            segflux = aperture_photometry(sourcemask, aper)['aperture_sum'][0]
            if (segflux == 0.):
                selected_pos.append((yi,xi))
    print('Number of available aperture positions: '+str(len(selected_pos)))

    # Random sampling
    sample_pos = random.sample(selected_pos,naper)

    # Get aperture measurements
    apfluxes = np.zeros(len(sample_pos))
    bgfluxes = np.zeros(len(sample_pos))
    apstd = np.zeros(len(sample_pos))
    if errstats:
        aperrs = np.zeros(len(sample_pos))

    for ii in range(len(sample_pos)):
        aper = CircularAperture(sample_pos[ii],aprad)
        ann  = CircularAnnulus(sample_pos[ii],r_in=aprad+1,r_out=aprad+1+bgdr)
        apfluxes[ii] = aperture_photometry(sub, aper)['aperture_sum'][0]
        bgfluxes[ii] = aperture_photometry(sub, ann)['aperture_sum'][0]/ann.area*aper.area
        apstd[ii] = ApertureStats(sub,aper).std
        if errstats:
            aperrs[ii] = np.sqrt(aperture_photometry(err**2, aper)['aperture_sum'][0])

    apfluxes_nobg = apfluxes - bgfluxes
    apstd_nocor = apstd*np.sqrt(aper.area)
    print('Aperture and background area (pixels): '+str(round(aper.area))+', '+str(round(ann.area)))

    # Get stats on full sky
    skyvalues = ima[(sourcemask==0)&(mask==0)]
    sky_mean, sky_std = avsigclip(skyvalues,3,3)
    if errstats:
        errvalues = err[(sourcemask==0)&(mask==0)]
        snrvalues = skyvalues/errvalues
        skysnr_mean, skysnr_std = avsigclip(snrvalues,3,3)

    # compute statistics
    mean_aper, std_aper = avsigclip(apfluxes_nobg,3,3)
    meanstd_stats, stdstd_stats = avsigclip(apstd_nocor,3,3)
    medianstd_stats = np.median(apstd_nocor)
    if errstats:
        meanerr_aper, stderr_aper = avsigclip(aperrs,3,3)

    rr = pixfrac/pscale*natscale
    if (rr >= 1.):
        drizcor = rr/(1-1/(3*rr)) # drizzling corr factor from Fruchter&Hook 
    else:
        drizcor = 1/(1-rr/3)
    print(f'Pixel correlation noise factor for pixfrac = {pixfrac}, pixel scale = {pscale} arcsec: {round(drizcor, 3)}')

    # get limiting flux and magnitude 
    apcor_at_limrad = get_apcor(apdiam_lim/2, apcorfname, natscale, filter, plot=False)
    print(f'Aperture correction at radius = {apdiam_lim/2}: {round(apcor_at_limrad,3)}')

    npix_aper = np.sqrt(aper.area)
    npix_lim = np.sqrt(np.pi*(apdiam_lim/2/pscale)**2)
    limflux_1sig_perpix = std_aper/npix_aper

    # calibrated 5 sigma flux, without aper correction
    #limflux_5sig_aper = 5 * npix_lim * limflux_1sig_perpix * 23.5045 * pscale**2 # muJy
    limflux_5sig_aper = 5 * npix_lim * limflux_1sig_perpix * calfactor # muJy
    limmag_AB_5sig = -2.5 * np.log10(limflux_5sig_aper * 1e-6) + 8.9

    # with aperture correction
    #limflux_5sig_apcor = 5 * npix_lim * limflux_1sig_perpix * apcor_at_limrad * 23.5045 * pscale**2 # muJy
    limflux_5sig_apcor = 5 * npix_lim * limflux_1sig_perpix * apcor_at_limrad * calfactor # muJy

    limmag_AB_5sig_apcor = -2.5 * np.log10(limflux_5sig_apcor * 1e-6) + 8.9
   
    print('*******************************************************')
    print('* Results:')
    print('*******************************************************')
    print(f'5 sigma limiting flux for a {apdiam_lim} arcsec diameter aperture: '+str(round(limflux_5sig_aper,6))+' muJy')
    print(f'5 sigma limiting AB mag for a {apdiam_lim} arcsec diameter aperture: '+str(round(limmag_AB_5sig,3)))
    print(f'5 sigma limiting AB mag (aperture corrected) for a {apdiam_lim} arcsec diameter aperture: '+str(round(limmag_AB_5sig_apcor,3)))
    print(f'Std per pixel from apertures: {round(limflux_1sig_perpix* 23.5045 * pscale**2,6)} muJy')
    print(f'Median std of sky pixels inside sky apertures: {round(medianstd_stats/npix_aper* 23.5045 * pscale**2,6)} muJy')
    print('Ratio of aper std to sky pixel std (compare with pixel correlation noise factor above):'+str(round(std_aper/medianstd_stats,2)))
    if errstats:
            print('Ratio of aper std to ERR extension aper mean: '+str(round(std_aper/meanerr_aper,3)))

    if plots:
        figapers,axs=plt.subplots(ncols=2,nrows=1,figsize=(14,10))
        axs[0].imshow(sourcemask, origin='lower',vmin=0, vmax=0.1, cmap='gray')
        axs[1].imshow(sub, origin='lower',vmin=0, vmax=0.1, cmap='plasma')
        patches = [plt.Circle(center, aprad_seg) for center in sample_pos]
        coll = matplotlib.collections.PatchCollection(patches, ec='red', fc='None')
        coll2 = copy(coll)
        axs[0].add_collection(coll)
        axs[1].add_collection(coll2)
        figapers.savefig(outroot+'_skyapers.png', bbox_inches='tight')

        if errstats:
            figsnr,ax = plt.subplots(figsize=(8,8))
            ax.hist(snrvalues,bins=100,color='tab:red',alpha=0.5,histtype='stepfilled', label='SNR (per pixel)')
            ax.hist(snrvalues,bins=100, histtype='step', color='maroon')
            meandiff, stddiff = avsigclip(snrvalues,3,3)
            ax.errorbar(meandiff,50000,marker='None',xerr=stddiff, color='k', lw=2, capsize=5, capthick=2, label='SNR stdev = '+str(round(stddiff,2)))
            ax.axvline(meandiff, ls=':',lw=2,color='k',label='Mean SNR = '+str(round(meandiff,2)))
            ax.set_xlim(meandiff-stddiff*4,meandiff+stddiff*4)
            ax.legend()
            figsnr.savefig(outroot+'_snrdist_errext.png',dpi=200,bbox_inches='tight')

        figaperdist,ax = plt.subplots(figsize=(8,8))
        ax.hist(apfluxes_nobg,bins=int(naper/25),color='tab:red',alpha=0.5,histtype='stepfilled', label='Aperture fluxes')
        ax.hist(apfluxes_nobg,bins=int(naper/25), histtype='step', color='maroon')
        halfy = ax.get_ylim()[1]/2
        ax.errorbar(0,halfy,marker='None',xerr=std_aper, color='k', lw=2, capsize=5, capthick=2, 
                    label='Aperture std = '+str(round(std_aper,6))+' '+bunit)
        ax.errorbar(0,(halfy-halfy/14),marker='None',xerr=medianstd_stats, color='tab:cyan', lw=2, capsize=5, 
                    capthick=2,label='pix2pix std = '+str(round(medianstd_stats,6))+' '+bunit)
        ax.errorbar(0,(halfy-2*halfy/14),marker='None',xerr=medianstd_stats*drizcor, color='tab:green', 
                    lw=2, capsize=5, capthick=2, label='pix2pix std (drizcor)')
        ax.errorbar(0,(halfy-3*halfy/14),marker='None',xerr=meanerr_aper, color='tab:purple', lw=2, 
                    capsize=5, capthick=2, label='ERR ext. std = '+str(round(meanerr_aper,6))+' '+bunit)
        ax.set_xlim(-std_aper*4,std_aper*4)
        ax.legend(fontsize='12', loc=3)
        ax.set_xlabel(f'F ({bunit})',size=14)
        ax.set_ylabel('N',size=14)
        ax.set_title(f'{detector}, {filter}')
        figaperdist.savefig(outroot+'_pixelstats_comp.png', dpi=200,bbox_inches='tight')

    return filter, limmag_AB_5sig, limmag_AB_5sig_apcor, std_aper/medianstd_stats, std_aper/meanerr_aper