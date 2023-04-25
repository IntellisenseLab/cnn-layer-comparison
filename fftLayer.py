import tensorflow

class FFTLayer2D(tensorflow.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False):
        super(FFTLayer2D, self).__init__(name=name, trainable=trainable, dynamic=dynamic, dtype=dtype)

    def call(self, input_image):
        self.image  = tensorflow.image.rgb_to_grayscale(input_image)
        self.imageC = tensorflow.dtypes.cast(self.image, tensorflow.complex64)
        self.fft    = tensorflow.signal.fft2d(self.imageC)

        self.fft_real = tensorflow.math.real(self.fft)
        self.fft_imag = tensorflow.math.imag(self.fft)

        self.fft_real_sq = tensorflow.math.square(self.fft_real)
        self.fft_imag_sq = tensorflow.math.square(self.fft_imag)

        self.fft_mag_sq = tensorflow.math.add(self.fft_real_sq, self.fft_imag_sq)

        self.magnitude = tensorflow.math.sqrt(self.fft_mag_sq)

        self.reduced_mag = tensorflow.math.log(self.magnitude)

        self.min = tensorflow.math.reduce_min(self.reduced_mag, axis=None, keepdims=False)
        self.max = tensorflow.math.reduce_max(self.reduced_mag, axis=None, keepdims=False)

        self.range = tensorflow.math.subtract(self.max, self.min)

        return tensorflow.math.divide_no_nan(self.reduced_mag, self.range)