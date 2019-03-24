import skimage.io as skimio
from skimage import img_as_float
import os
import sys
import numpy as np
import time
from skimage.color import rgb2gray

# local
from model import build_generator
from contrast_stretch import contrast_stretch

# for running using CPU only
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: python dehaze.py <hazy_images_path> <output_path>'
        print 'falling back to default directories'
        hazy_path = './hazy_images'
        out_path = './out'
    else:
        hazy_path = sys.argv[1]
        out_path = sys.argv[2]
        
    wt_path = './model_wt/gen_wt.h5'

    hazy_files = sorted(os.listdir(hazy_path))
    enh_patch_size = [128]

    # model
    model = build_generator((1200, 1600, 3))
    model.load_weights(wt_path)

    for i in xrange(len(hazy_files)):
        sys.stdout.write('[{}/{}] - {}'.format(i+1, len(hazy_files),
                                               hazy_files[i]))
        sys.stdout.flush()

        hazy_im = img_as_float(skimio.imread(os.path.join(hazy_path,
                                                          hazy_files[i])))
            
        start = time.time()
        im2 = rgb2gray(contrast_stretch(hazy_im,
                                        enh_patch_size[0]))[..., None]
            
        img_n = np.random.uniform(-1, 1, hazy_im.shape)
            
        out = model.predict([hazy_im[None, ...], im2[None, ...],
                             img_n[None, ...]])
        end = time.time()

        sys.stdout.write(' |time: {} s\n'.format(end - start))
        sys.stdout.flush()
        out_name = os.path.splitext(hazy_files[i])[0]
        skimio.imsave(os.path.join(out_path, out_name+'_out.png'), out[0])
