from keras.layers import Concatenate, Lambda, Input
from keras.layers import Conv2D, UpSampling2D
from keras import layers
from keras.models import Model
from keras.backend import tf as ktf
import numpy as np

from keras_contrib.layers.normalization import InstanceNormalization


def build_generator(in_shape):
    inp = Input(in_shape)
    
    resized_shape = np.divide(in_shape[:-1], 4)
    resized_in = Lambda(lambda x:
                        ktf.image.resize_bilinear(x, resized_shape),
                        name='resized_im')(inp)
    
    contrast_in = Input((in_shape[0], in_shape[1], 1))
    noise_in = Input(in_shape)

    # resized im path
    rx = Conv2D(8, (3, 3), activation='elu', padding='same')(resized_in)
    rx = residual_block_m(rx, 16, _project_shortcut=True)
    rx = residual_block_m(rx, 16)

    # original image path
    ox = Conv2D(8, (3, 3), activation='elu', padding='same')(inp)
    ox = residual_block_m(ox, 8, _project_shortcut=True)
    ox = residual_block_m(ox, 16, _strides=(2, 2))
    ox = residual_block_m(ox, 32, _strides=(2, 2))

    mx = Concatenate(axis=-1)([ox, rx])
    mx = UpSampling2D((2, 2))(mx)
    mx = residual_block_m(mx, 32, _project_shortcut=True)
    mx = UpSampling2D((2, 2))(mx)
    mx = residual_block_m(mx, 32)
    
    # contrast image path
    cx = Conv2D(8, (3, 3), activation='elu', padding='same')(contrast_in)
    cx = residual_block_m(cx, 16, _project_shortcut=True)

    # noise image path
    nx = Conv2D(8, (3, 3), activation='elu', padding='same')(noise_in)
    nx = residual_block_m(nx, 16, _project_shortcut=True)
    nx = residual_block_m(nx, 16)

    x = Concatenate(axis=-1)([mx, cx, nx])
    x = Conv2D(16, (3, 3), padding='same', activation='elu')(x)
    x = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)

    gen = Model(inputs=[inp, contrast_in, noise_in], outputs=[x])

    return gen


###########################
# Source: https://gist.github.com/mjdietzx/5319e42637ed7ef095d430cb5c5e8c64
# then modified
##########################
def residual_block_m(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = InstanceNormalization(axis=-1)(y)
    y = layers.ELU()(y)
    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides,
                      padding='same')(y)

    y = InstanceNormalization(axis=-1)(y)
    y = layers.ELU()(y)
    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1),
                      padding='same')(y)
    
    # identity shortcuts used directly when the input and
    # output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to
        # match dimensions (done by 1x1 convolutions)
        # when the shortcuts go across feature maps of two sizes,
        # they are performed with a stride of 2
        shortcut = InstanceNormalization(axis=-1)(shortcut)
        shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1),
                                 strides=_strides, padding='same')(shortcut)

    y = layers.add([shortcut, y])

    return y


if __name__ == '__main__':
    m = build_generator((1200, 1600, 3))

    m.summary()

    from keras.utils import plot_model
    plot_model(m, show_shapes=True, to_file='gen.png')
