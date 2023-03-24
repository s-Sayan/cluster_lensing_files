import numpy as np 
import camb 
from numpy.fft import fftshift
from tqdm import tqdm

import lensit as li
from lensit.clusterlens import lensingmap, profile 
from lensit.misc.misc_utils import gauss_beam
from lensit.ffs_covs import ffs_cov, ell_mat
from plancklens.wigners import wigners

import os
import os.path as op
import matplotlib as mpl
from matplotlib import pyplot as plt

from scipy.interpolate import UnivariateSpline as spline

cambinifile = 'planck_2018_acc'

pars = camb.read_ini(op.join(op.dirname('/Users/sayan/CMB_WORK/CAMB-1.1.3/inifiles'),  'inifiles', cambinifile + '.ini'))
results = camb.get_results(pars)

# We define here the parameters for the profile of the cluster
M200, z = 2 * 1e14, 0.7
profname = 'nfw'
key = "lss" # "lss"/"cluster"/"lss_plus_cluster"
profparams={'M200c':M200, 'z':z}

# Define here the map square patches
npix = 1024  # Number of pixels
lpix_amin = 0.3 # Physical size of a pixel in arcmin (There is bug when <0.2 amin, due to low precision in Cl_TE at )

print("The size of the data patch is %0.1f X %0.1f arcmin central box"%(npix*lpix_amin, npix*lpix_amin))

# Maximum multipole used to generate the CMB maps from the CMB power spectra
# ellmaxsky = 6000 # (bug when ellmax>6300 because of low precision in Cl_TE of CAMB )
ellmaxsky = 8000 

# Set the maximum ell observed in the CMB data maps
ellmaxdat = 4000
ellmindat = 100

# Number of simulated maps 
nsims = 1000

# Set CMB experiment for noise level and beam
cmb_exp='S4_sayan'

# We will cache things in this directory 
libdir = lensingmap.get_cluster_libdir(cambinifile,  profname, key, npix, lpix_amin, ellmaxsky, M200, z, nsims, cmb_exp)
#libdir = op.join(libdir,"trunc")
print(libdir)

lmax = ellmaxsky
cpp_fid = results.get_lens_potential_cls(lmax=lmax, raw_cl=True).T[0]

camb_cls = results.get_unlensed_scalar_cls(CMB_unit='muK', raw_cl=True, lmax=lmax).T
cls_unl_fid = {'tt':camb_cls[0], 'ee':camb_cls[1], 'bb':camb_cls[2], 'te':camb_cls[3], 'pp':cpp_fid}

camb_cls_len = results.get_lensed_scalar_cls(CMB_unit='muK', raw_cl=True, lmax=lmax).T
cls_len_fid = {'tt':camb_cls_len[0], 'ee':camb_cls_len[1], 'bb':camb_cls_len[2], 'te':camb_cls_len[3], 'pp':cpp_fid}

camb_cls_len = results.get_lensed_gradient_cls(CMB_unit='muK', raw_cl=True, lmax=lmax).T
cls_gradlen_fid = {'tt':camb_cls_len[0], 'ee':camb_cls_len[1], 'bb':camb_cls_len[2], 'te':camb_cls_len[4], 'pp':cpp_fid}

np.random.seed(seed=20)
clustermaps = lensingmap.cluster_maps(libdir, key, npix, lpix_amin, nsims, results, profparams, profilename=profname,  ellmax_sky = ellmaxsky, ellmax_data=ellmaxdat, ellmin_data=ellmindat, cmb_exp=cmb_exp)

def cli(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret

def pp_to_kk(ls):
    return ls ** 2 * (ls+1.) ** 2 * 0.25 

def p_to_k(ls):
    return ls * (ls+1.) * 0.5

def kk_to_pp(ls):
    return cli(pp_to_kk(ls))

def k_to_p(ls):
    return cli(p_to_k(ls))

def th_amin_to_el(th_amin):
    th_rd = (th_amin/60)*(np.pi/180)
    return np.pi/th_rd

ellmax_sky = clustermaps.ellmax_sky
sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax = li.get_config(clustermaps.cmb_exp)

cls_noise = {'t': (sN_uKamin * np.pi / 180. / 60.) ** 2 * np.ones(clustermaps.ellmax_sky + 1),
            'q':(sN_uKaminP * np.pi / 180. / 60.) ** 2 * np.ones(clustermaps.ellmax_sky + 1),
            'u':(sN_uKaminP * np.pi / 180. / 60.) ** 2 * np.ones(clustermaps.ellmax_sky + 1)}  # simple flat noise Cls
# cl_transf = gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=ellmax_sky)
# lib_alm = ell_mat.ffs_alm_pyFFTW(get_ellmat(LD_res, HD_res=HD_res),
                    # filt_func=lambda ell: (ell >= ellmin) & (ell <= ellmax), num_threads=pyFFTWthreads)
# lib_skyalm = ell_mat.ffs_alm_pyFFTW(clustermaps.ellmat,
                    # filt_func=lambda ell: (ell <= ellmax_sky), num_threads=clustermaps.num_threads)

cl_transf = clustermaps.cl_transf
lib_skyalm = clustermaps.lib_skyalm


typ = 'T'

lib_dir = op.join(clustermaps.dat_libdir, typ)
# isocov = ffs_cov.ffs_diagcov_alm(lib_dir, clustermaps.lib_datalm, clustermaps.cls_unl, cls_len, cl_transf, cls_noise, lib_skyalm=lib_skyalm)
isocov = ffs_cov.ffs_diagcov_alm(lib_dir, clustermaps.lib_datalm, cls_unl_fid, cls_len_fid, cl_transf, cls_noise, lib_skyalm=lib_skyalm)

ell, = np.where(lib_skyalm.get_Nell()[:ellmaxsky+1])
cpp_prior =  cpp_fid

def get_starting_point(idx, typ, clustermaps): 
    """
    This returns initial data for simulation index 'idx' from a CMB-S4 simulation library.
    On first call the simulation library will generate all simulations phases, hence might take a little while.
    """ 

    print(" I will be using data from ell=%s to ell=%s only"%(isocov.lib_datalm.ellmin, isocov.lib_datalm.ellmax))
    print(" The sky band-limit is ell=%s"%(isocov.lib_skyalm.ellmax))
    # isocov.lib_datalm defines the mode-filtering applied on the data, 
    # and isocov.lib_skyalm the band-limits of the unlensed sky.
    lib_qlm = lib_skyalm #: This means we will reconstruct the lensing potential for all unlensed sky modes.
    ellmax_sky = lib_skyalm.ellmax
    ell = np.arange(ellmax_sky+1)
    # lib_qlm = isocov.lib_datalm #: This means we will reconstruct the lensing potential for data modes.

    # We now build the Wiener-filtered quadratic estimator. We use lensed CMB spectra in the weights.
    if typ=='QU':
        datalms = np.array([isocov.lib_datalm.map2alm(m) for m in clustermaps.maps_lib.get_sim_qumap(idx)]) 
    elif typ =='T':
        datalms = np.array([isocov.lib_datalm.map2alm(clustermaps.maps_lib.get_sim_tmap(idx))]) 
    elif typ =='TQU':
        datalms = np.array([isocov.lib_datalm.map2alm(m) for m in np.array([clustermaps.maps_lib.get_sim_tmap(idx), clustermaps.maps_lib.get_sim_qumap(idx)[0], clustermaps.maps_lib.get_sim_qumap(idx)[1]])]) 
    
    use_cls_len = True
    
    #H0len1 =  cli(isocov.get_N0cls(typ, lib_qlm, use_cls_len=use_cls_len)[0])
    H0len =  isocov.get_response(typ, lib_qlm, cls_weights=cls_gradlen_fid, use_cls_len=use_cls_len)[0]
    plm = 0.5 * isocov.get_qlms(typ,  isocov.get_iblms(typ, datalms, use_cls_len=use_cls_len)[0], lib_qlm, 
                                 use_cls_len=use_cls_len)[0]
    
    # Normalization and Wiener-filtering:
    # cpp_prior = li.get_fidcls()[0]['pp'][:lib_qlm.ellmax+1]

    plmqe  = lib_qlm.almxfl(plm, cli(H0len), inplace=False)
    klmqe  = lib_qlm.almxfl(plmqe, p_to_k(ell), inplace=False)
    plm0  = lib_qlm.almxfl(plm, cli(H0len + cli(cpp_prior[:lib_qlm.ellmax+1])), inplace=False)
    klm0  = lib_qlm.almxfl(plm0, p_to_k(ell), inplace=False)
    #klm0 = lib_qlm.almxfl(klmqe, ckk_prior[:lib_skyalm.ellmax+1]*cli(cli(H0len)*pp_to_kk(ell) + ckk_prior[:lib_skyalm.ellmax+1]), inplace=False)
    wf_qe = np.zeros(lib_qlm.ellmax+1)
    ell, = np.where(lib_qlm.get_Nell()[:isocov.lib_datalm.ellmax])
    
    wf_qe[ell] = cpp_prior[ell] * cli(cpp_prior[ell] + cli(H0len[ell]))

    # wf_qe = cpp_prior[:lib_qlm.ellmax+1] * cli(cpp_prior[:lib_qlm.ellmax+1] + cli(H0len[:lib_qlm.ellmax+1]))
    # wf_qe = spline(ell, wf_qe_0[ell])(np.arange(lib_qlm.ellmax+1))


    # Initial likelihood curvature guess. We use here N0 as calculated with unlensed CMB spectra:
    #H0unl =  cli(isocov.get_N0cls(typ, lib_qlm, use_cls_len=False)[0])
    return plm0, plmqe, klm0, klmqe, H0len, wf_qe, datalms


idx = 0
plm0, plmqe, klm0, klmqe, H0len, wf_qe, datalms= get_starting_point(idx, typ, clustermaps)

nmaps = 1000
plm0s = [None]*nmaps
plmqes = [None]*nmaps
datalms = [None]*nmaps
plm_input = [None]*nmaps
cross_cor = [None]*nmaps
auto_cor = [None]*nmaps
#len_map_cl = [None]*nmaps
#noise_map_cl = [None]*nmaps

if nsims >1:
    for idx in range(nmaps):
        print(idx)
        plm0s[idx], plmqes[idx],  klm0, klmqe, H0len, wf_qe, datalms[idx] = get_starting_point(idx, typ, clustermaps)
        plm_input[idx] = lib_skyalm.map2alm(clustermaps.len_cmbs._get_f(idx).get_phi())
        plmplm1 = np.multiply(plm_input[idx], plmqes[idx].conjugate())
        plmplm2 = np.multiply(plmqes[idx], plmqes[idx].conjugate())
        cross_cor[idx] = lib_skyalm.bin_realpart_inell(plmplm1)
        auto_cor[idx] = lib_skyalm.bin_realpart_inell(plmplm2)
        
cross_cor_av = np.mean(cross_cor, axis=0)
auto_cor_av = np.mean(auto_cor, axis=0) 

from plancklens import n0s
N0s, N0s_curl = n0s.get_N0(beam_fwhm= 1.,nlev_t= 1.,nlev_p=1*np.sqrt(2.), lmin_CMB=ellmindat, lmax_CMB=ellmaxdat, lmax_out=lmax, cls_len=cls_len_fid,cls_glen=cls_gradlen_fid, cls_weight=cls_len_fid)

##Plotting
ell, = np.where(lib_skyalm.get_Nell()[:ellmaxdat])
el = ell[1:]
plt.plot(el, (cross_cor_av[el]-cpp_fid[el])/cpp_fid[el], label="fractional difference in cross correlation")
plt.axhline(0, c="orange")
plt.legend()
plt.title("with lensed grad Cl")
plt.savefig("cross_correlation_QE.pdf")

plt.plot(el, (auto_cor_av[el] - (N0s["ptt"][el] + cpp_fid[el]))/(N0s["ptt"][el] + cpp_fid[el]), label="fractional difference in auto correlation")
plt.axhline(0, c="orange")
plt.legend()
plt.title("with lensed grad Cl")
plt.savefig("auto_correlation_QE.pdf")