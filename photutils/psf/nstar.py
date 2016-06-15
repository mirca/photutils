from astropy.table import Table, vstack
import numpy as np
from .daogroup import daogroup
from .core import subtract_psf
from .core import _call_fitter
import matplotlib.pyplot as plt

def nstar(image, groups, shape, fitter, psf_model, weights=None,
          plot_regions=False, **psf_kwargs):
    """
    Fit, as appropriate, a compound or single model to the given `groups` of
    stars. Groups are fitted sequentially from the smallest to the biggest. In
    each iteration, `image` is subtracted by the previous fitted group. 
    
    Parameters
    ----------
    image : numpy.ndarray
        Background-subtracted image.
    groups : list of `~astropy.table.Table`
        Each `~astropy.table.Table` in this list corresponds to a group of
        mutually overlapping starts.
    shape : tuple
        Shape of a rectangular region around the center of an isolated source.
    fitter : `~astropy.modeling.fitting.Fitter` instance
        An instance of an `~astropy.modeling.fitting.Fitter`
        See `~astropy.modeling.fitting` for details about fitters.
    psf_model : `~astropy.modeling.Fittable2DModel` 
        The PSF/PRF analytical model. This model must have centroid and flux
        as parameters.
    weights : numpy.ndarray
        Weights used in the fitting procedure.
    psf_kwargs : dict
        Fixed parameters to be passed to `psf_model`.
    plot_regions : boolean
        If True, plot the regions, which were used to fit each group, to the
        current gca.

    Return
    ------
    result_tab : `~astropy.table.Table`
        Astropy table that contains the results of the photometry.
    image : numpy.ndarray
        Residual image.
    """
    
    result_tab = Table([[], [], [], [],],
                       names=('id', 'x_fit', 'y_fit', 'flux_fit'),
                       dtype=('i4', 'f8', 'f8', 'f8'))
    models_order = _get_models_order(groups) 
    while len(models_order) > 0:
        curr_order = np.min(models_order)
        n = 0
        N = len(models_order)
        while(n < N):
            if curr_order == len(groups[n]):
                group_psf = _get_group_psf(psf_model, groups[n], **psf_kwargs)
                x, y, data = _extract_shape_and_data(shape, groups[n], image)
                fitted_model = _call_fitter(fitter, group_psf, x, y, data,
                                            weights)
                param_table = _model_params_to_table(fitted_model, groups[n])
                result_tab = vstack([result_tab, param_table])
                # image = subtract_psf(image, psf_model(**psf_kwargs),
                #                      param_table)
                image = _subtract_psf(image, x, y, fitted_model)
                models_order.remove(curr_order)
                del groups[n]
                N = N - 1
                if plot_regions: 
                    patch = _show_region([(np.min(x), np.min(y)),
                                          (np.min(x), np.max(y)),
                                          (np.max(x), np.max(y)),
                                          (np.max(x), np.min(y)),
                                          (np.min(x), np.min(y)),])
                    plt.gca().add_patch(patch)
            n = n + 1
    return result_tab, image


def _model_params_to_table(fitted_model, group):
    """
    Place fitted parameters into an astropy table.
    
    Parameters
    ----------
    fitted_model : Fittable2DModel
    group : ~astropy.table.Table
    
    Returns
    -------
    param_tab : ~astropy.table.Table
        Table that contains the fitted parameters.
    """

    param_tab = Table([[],[],[],[]], names=('id','x_fit','y_fit','flux_fit'),
                      dtype=('i4','f8','f8','f8'))
    if np.size(fitted_model) == 1:
        tmp_table = Table([[group['id'][0]],
                           [getattr(fitted_model,'x_0').value],
                           [getattr(fitted_model, 'y_0').value],
                           [getattr(fitted_model, 'flux').value]],
                           names=('id','x_fit', 'y_fit', 'flux_fit'))
        param_tab = vstack([param_tab, tmp_table])
    else:
        for i in range(np.size(fitted_model)):
            tmp_table = Table([[group['id'][i]],
                               [getattr(fitted_model,'x_0_'+str(i)).value],
                               [getattr(fitted_model, 'y_0_'+str(i)).value],
                               [getattr(fitted_model, 'flux_'+str(i)).value]],
                               names=('id','x_fit', 'y_fit', 'flux_fit'))
            param_tab = vstack([param_tab, tmp_table])

    return param_tab


def _get_group_psf(psf_model, group, **psf_fixed_params):
    """
    This function computes the sum of PSFs with model given by `psf_model` and
    (x_0, y_0, flux_0) given by `group` as a astropy compound model.

    Parameters
    ----------
    psf_model : `~astropy.modeling.Fittable2DModel`
        The PSF/PRF analytical model. This model must have centroid and flux
        as parameters.
    group : `~astropy.table.Table`
        Table from which the compound PSF/PRF will be generated.
        It must have columns named as `x_0`, `y_0`, and `flux_0`.
    psf_fixed_params : dict
        Fixed parameters to be passed to `psf_model`.
    
    Returns
    -------
    group_psf : CompoundModel 
        CompoundModel as the sum of the PSFs/PRFs models.

    See
    ---
    `~daogroup`
    """
    
    group_psf = psf_model(flux=group['flux_0'][0], x_0=group['x_0'][0],
                          y_0=group['y_0'][0], **psf_fixed_params)
    for i in range(len(group) - 1):
        group_psf += psf_model(flux=group['flux_0'][i+1], x_0=group['x_0'][i+1],
                               y_0=group['y_0'][i+1], **psf_fixed_params)
    return group_psf


def _extract_shape_and_data(shape, group, image):
    """
    Parameters
    ----------
    shape : tuple
        Shape of a rectangular region around the center of an isolated source.
    group : `astropy.table.Table`
        Group of stars
    image : numpy.ndarray

    Returns
    -------
    x, y : numpy.mgrid
        All coordinate pairs (x,y) in a rectangular region which encloses all
        sources of the given group
    image : numpy.ndarray
        Pixel value
    """

    xmin = int(np.around(np.min(group['x_0'])) - shape[0])
    xmax = int(np.around(np.max(group['x_0'])) + shape[0])
    ymin = int(np.around(np.min(group['y_0'])) - shape[1])
    ymax = int(np.around(np.max(group['y_0'])) + shape[1])
    y,x = np.mgrid[ymin:ymax+1, xmin:xmax+1]

    return x, y, image[ymin:ymax+1, xmin:xmax+1]


def _get_models_order(groups):
    """
    Parameters
    ----------
    groups : list
        List of groups of mutually overlapping stars.

    Returns
    -------
    model_order : list
        List with the number of stars per group.
    """

    model_order = []
    for i in range(len(groups)):
        model_order.append(len(groups[i]))
    return model_order


def _show_region(verts):
    from matplotlib.path import Path
    import matplotlib.patches as patches

    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO,
             Path.CLOSEPOLY,]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor="none", lw=1)
    return patch

# No need for this. Should use photutils.psf.subtract_psf instead
def _subtract_psf(image, x, y, fitted_model):
    psf_image = np.zeros(image.shape)
    psf_image[y,x] = fitted_model(x,y)
    return image - psf_image
