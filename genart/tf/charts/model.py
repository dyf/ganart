import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, BatchNormalization, ReLU, LeakyReLU, Dropout, Input, Activation, ZeroPadding2D, Concatenate, Flatten, Reshape, Conv2DTranspose, Embedding, UpSampling2D, AveragePooling2D, Add
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential


class WeightedSum(Add):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = tf.keras.backend.variable(0.0, name='weighted_sum_alpha')

    def _merge_function(self, inputs):
        return (1.0-self.alpha) * inputs[0] + self.alpha * inputs[1]

class PixelNormalization(Layer):
	# initialize the layer
	def __init__(self, **kwargs):
		super(PixelNormalization, self).__init__(**kwargs)
 
	# perform the operation
	def call(self, inputs):
		# calculate square pixel values
		values = inputs**2.0
		# calculate the mean pixel values
		mean_values = tf.keras.backend.mean(values, axis=-1, keepdims=True)
		# ensure the mean is not zero
		mean_values += 1.0e-8
		# calculate the sqrt of the mean squared value (L2 norm)
		l2 = tf.keras.backend.sqrt(mean_values)
		# normalize values by the l2 norm
		normalized = inputs / l2
		return normalized
 
	# define the output shape of the layer
	def compute_output_shape(self, input_shape):
		return input_shape

def gen_conv_block(nf, name=None):
     return Sequential([
        Conv2D(nf, (3,3), strides=(1,1), padding='same'),
        #BatchNormalization(),
        PixelNormalization(),
        LeakyReLU()                 
     ], name=name)

def disc_conv_block(nf, name=None):
    return Sequential([
        Conv2D(nf, (3,3), strides=(1,1), padding='same'),        
        LeakyReLU()
    ], name=name)

def gen_layer_bc(nf, name=None):
    return Sequential([
        UpSampling2D(),
        gen_conv_block(nf),
        gen_conv_block(nf)
    ], name=name)

def gen_layer_tc(nf, k=(5,5), strides=(2,2), name=None):
    return Sequential([
        Conv2DTranspose(nf, k, strides=strides, padding='same'), 
        BatchNormalization(),
        LeakyReLU(),
    ], name=name)

def disc_layer_dpool(nf, name=None):
    return Sequential([
        disc_conv_block(nf),
        disc_conv_block(nf),
        AveragePooling2D()
    ], name=name)

def disc_layer_sc(nf, name=None):
    return Sequential([
        Conv2D(nf, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(),
        Dropout(0.3),
    ], name=name)

class PCGANBuilder:
    def __init__(self, latent_size=256, num_labels=8, max_shape=(512,512,3)):
        self.latent_size = latent_size
        self.num_labels = num_labels
        self.max_shape = max_shape

    def build_model(self, layer_sizes=[70,60,50,40,30,20,10]):
        return ( 
            self.build_generator(layer_sizes), 
            self.build_discriminator(layer_sizes[::-1])
        )
        
    def build_generator(self, layer_sizes):        

        num_models = len(layer_sizes)        

        factor = pow(2, num_models-1)
        image_shape = ( self.max_shape[0]//factor, self.max_shape[1]//factor, self.max_shape[2] )

        labels_input = Input(shape=(self.num_labels,), name='gen_labels_input')
        
        labels = Sequential([
            Dense(image_shape[0]*image_shape[1]),
            Reshape([image_shape[0], image_shape[1], 1])
        ], name='gen_labels_reshape')(labels_input)

        latent_input = Input(shape=(self.latent_size,), name='gen_latent_input')
        latent = Sequential([
                Dense(image_shape[0]*image_shape[1]*(layer_sizes[0]-1)),
                BatchNormalization(),
                LeakyReLU(),
                Reshape((image_shape[0],image_shape[1],(layer_sizes[0]-1)))
        ], name='gen_latent_reshape')(latent_input)

        gen = Concatenate()([labels, latent])
        gen = gen_conv_block(layer_sizes[0], name='gen_conv_0')(gen)
        
        models = []
        for mi in range(num_models):
            mgen = gen
            
            prev_layers = build_layer_dict(models[-1]) if mi > 0 else {}
            
            prev_out = None

            if mi > 0:
                prev_layers_to_apply = [ f'gen_conv_{i}' for i in range(mi) ]
                
                for layer_name in prev_layers_to_apply:
                    prev_layer = prev_layers[layer_name]
                    mgen = prev_layer(mgen)
                prev_out = mgen
            
            if mi > 0:                
               mgen = gen_layer_bc(layer_sizes[mi], name=f"gen_conv_{mi}")(mgen)            
            
            mgen = Conv2D(3, (1,1), padding='same', activation='tanh', name=f'gen_to_rgb_level_{mi}')(mgen) # 512

            if mi > 0:
                upgen = Sequential([
                    prev_layers[f'gen_to_rgb_level_{mi-1}'],
                    UpSampling2D()
                ], name=f'gen_upsample_level_{mi}')(prev_out)

                mgen = WeightedSum(name=f'gen_ws_level_{mi}')([upgen, mgen])

            model = Model([labels_input, latent_input], mgen)
            models.append(model)

            
        return models

    def build_discriminator(self, layer_sizes):
        num_models = len(layer_sizes)

        labels_input = Input(shape=(self.num_labels,), name=f'disc_labels_input')
        image_inputs = []
        models = []

        for mi in range(num_models):    
            fidx = -mi-1

            factor = pow(2, num_models-mi-1)
            image_shape = ( self.max_shape[0]//factor, self.max_shape[1]//factor, self.max_shape[2] )

            image_input = Input(shape=image_shape, name=f'disc_image_input_level_{mi}')
            image_inputs.append(image_input)
                
            labels_reshape = Sequential([
                Dense(image_shape[0]*image_shape[1]),
                Reshape((image_shape[0], image_shape[1],1))
            ], name=f'disc_labels_reshape_level_{mi}')(labels_input)        

            disc = Concatenate(name=f'disc_incat_level_{mi}')([labels_reshape, image_input])
            disc = Sequential([
                Conv2D(layer_sizes[fidx], (1,1), strides=(1,1), padding='same', ),
                LeakyReLU()
            ], name=f'disc_from_rgb_level_{mi}')(disc) 

            if mi > 0:
                new_disc = disc_layer_dpool(layer_sizes[fidx+1], name=f'disc_down_{mi}')(disc)
                up_disc = Sequential([
                    AveragePooling2D(),
                    Conv2D(layer_sizes[fidx+1], (1,1), strides=(1,1)),
                    LeakyReLU()
                ], name=f'disc_pooldown_{mi}')(disc)

                disc = WeightedSum(name=f'weighted_sum_level_{mi}')([up_disc, new_disc])
                
                prev_layers_to_apply = [ f'disc_down_{i}' for i in range(mi-1,-1,-1) ]
                prev_layers = build_layer_dict(models[-1])

                for layer_name in prev_layers_to_apply:
                    prev_layer = prev_layers[layer_name]
                    disc = prev_layer(disc)
            else:
                disc = Sequential([
                    Flatten(),
                    Dense(1)#, activation='sigmoid')
                ], name=f'disc_down_{mi}')(disc)

            model = Model([labels_input, image_input], disc, name=f'disc_{image_shape[0]}')
            models.append(model)

        return models

def build_layer_dict(model):
    return dict([(layer.name, layer) for layer in model.layers])

def build_gan(latent_size=100):

    generator = Sequential([
        Dense(8*8*256, use_bias=False, input_shape=(latent_size,)),
        BatchNormalization(),
        LeakyReLU(),

        Reshape((8,8,256)),
        
        Conv2DTranspose(256, (5,5), strides=(1,1), padding='same', use_bias=False), # 8
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(256, (5,5), strides=(2,2), padding='same', use_bias=False), # 16
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False), # 32
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False), # 64
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False), # 128
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False), # 256
        BatchNormalization(),
        LeakyReLU(),
        
        Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh') # 512
    ])

    discriminator = Sequential([
        Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[512, 512, 3]),
        LeakyReLU(),
        Dropout(0.3),
        
        Conv2D(64, (5, 5), strides=(2, 2), padding='same'), 
        LeakyReLU(),
        Dropout(0.3),
        
        Conv2D(128, (5, 5), strides=(2, 2), padding='same'), 
        LeakyReLU(),
        Dropout(0.3),

        Conv2D(128, (5, 5), strides=(2, 2), padding='same'), 
        LeakyReLU(),
        Dropout(0.3),

        Conv2D(256, (5, 5), strides=(2, 2), padding='same'), 
        LeakyReLU(),
        Dropout(0.3),

        Conv2D(256, (5, 5), strides=(2, 2), padding='same'), 
        LeakyReLU(),
        Dropout(0.3),

        Flatten(),
        Dense(1)
    ])

    return generator, discriminator

if __name__ == "__main__":
    pcgan = PCGANBuilder()

    gen, disc = pcgan.build_model()
    for g in gen:
        print(g.summary())