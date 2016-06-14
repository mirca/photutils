from astropy.table import Table, vstack
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.stats import sigma_clipped_stats
from astropy.modeling.fitting import LevMarLSQFitter
from numpy.testing import assert_allclose
from ...datasets import make_gaussian_sources
from ...datasets import make_noise_image
from ..daogroup import daogroup
from ..nstar import nstar, _get_group_psf
from ..core import IntegratedGaussianPRF

class TestNSTAR(object):
    def test_nstar_one(self):
        """
        """
        sources = Table()
        sources['flux'] = [700, 700]
        sources['x_mean'] = [12, 17]
        sources['y_mean'] = [15, 15]
        sources['x_stddev'] = [2.0, 2.0]
        sources['y_stddev'] = sources['x_stddev']
        sources['theta'] = [0, 0]
        tshape = (32, 32)

        image = (make_gaussian_sources(tshape, sources) +
                 make_noise_image(tshape, type='poisson', mean=1.,
                                  random_state=1))
        
        _, bkg, std = sigma_clipped_stats(image, sigma=3.0, iters=5)

        sources['flux'].name = 'flux_0'
        sources['x_mean'].name = 'x_0'
        sources['y_mean'].name = 'y_0'

        groups = daogroup(sources, 2.0*1.5*gaussian_sigma_to_fwhm)
        result_tab, residual_img = nstar(image-bkg, groups, (5,5),
                                         LevMarLSQFitter(),
                                         IntegratedGaussianPRF, sigma=2.0)
        assert_allclose(result_tab['x_fit'], sources['x_0'], rtol=1e-2)
        assert_allclose(result_tab['y_fit'], sources['y_0'], rtol=1e-2)
        assert_allclose(result_tab['flux_fit'], sources['flux_0'], rtol=1e-1)

    def test_nstar_two(self):
        """
        """
        sources = Table()
        sources['flux'] = [500, 700, 500, 600]
        sources['x_mean'] = [12, 17, 12, 17]
        sources['y_mean'] = [15, 15, 20, 20]
        sources['x_stddev'] = [2.0, 2.0, 2.0, 2.0]
        sources['y_stddev'] = sources['x_stddev']
        sources['theta'] = [0, 0, 0, 0]
        tshape = (32, 32)

        image = (make_gaussian_sources(tshape, sources) +
                 make_noise_image(tshape, type='poisson', mean=1.,
                                  random_state=1234))
        
        _, bkg, std = sigma_clipped_stats(image, sigma=3.0, iters=5)

        sources['flux'].name = 'flux_0'
        sources['x_mean'].name = 'x_0'
        sources['y_mean'].name = 'y_0'

        groups = daogroup(sources, 2.0*1.5*gaussian_sigma_to_fwhm)
        result_tab, residual_img = nstar(image-bkg, groups, (5,5),
                                         LevMarLSQFitter(),
                                         IntegratedGaussianPRF, sigma=2.0)
        assert_allclose(result_tab['x_fit'], sources['x_0'], rtol=1e-2)
        assert_allclose(result_tab['y_fit'], sources['y_0'], rtol=1e-2)
        assert_allclose(result_tab['flux_fit'], sources['flux_0'], rtol=1e-1)
