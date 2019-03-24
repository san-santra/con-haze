from skimage import exposure
import numpy as np


def contrast_stretch(im, patch_size):
    '''
    local contrast stretch in each patch and aggregate
    '''
    (nrow, ncol, nch) = im.shape

    xrow = range(0, nrow - patch_size+1, 20)
    ycol = range(0, ncol - patch_size+1, 20)

    # add extra index to account for the border, can't just leave it
    if xrow[-1]+patch_size != nrow:
        xrow.append(nrow - patch_size)
        
    if ycol[-1]+patch_size != ncol:
        ycol.append(ncol - patch_size)

    agg_im = np.zeros(im.shape, dtype='float32')
    count = np.zeros((nrow, ncol), dtype='int')
    p_patch = np.zeros((patch_size, patch_size, 3), dtype='float32')
    
    for r in xrow:
        for c in ycol:
            patch = im[r:r+patch_size, c:c+patch_size, :]

            for i in xrange(nch):
                p_patch[:, :, i] = exposure.rescale_intensity(patch[:, :, i])

            agg_im[r:r+patch_size, c:c+patch_size, :] += p_patch
            count[r:r+patch_size, c:c+patch_size] += 1
    
    # aggregate
    out_im = agg_im/np.tile(count[:, :, None], (1, 1, 3))

    return out_im
