import tensorflow as tf
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.sample_batch import SampleBatch


class KerasModel(TFModelV2):
    """Custom model for PPO.
    It is made a convolutional neural network for analysing the graph image and then concatenates \\
    the output with the algorithm current open position to decide unpon a final action """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(KerasModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        
        # Input observations in the NN
        self.image = tf.keras.layers.Input(shape=original_space['image'].shape, name="image")
        self.position = tf.keras.layers.Input(shape=original_space['position'].shape, name="position")

        # Create CNN for graph image analysis
        conv1 = tf.keras.layers.Conv2D(4, (3, 3), (5, 5))(self.image)
        conv2 = tf.keras.layers.Conv2D(8, (2, 2), (4, 4))(conv1)
        conv3 = tf.keras.layers.Conv2D(16, (2, 2), (3, 3))(conv2)
        conv4 = tf.keras.layers.Conv2D(32, (2, 2), (2, 2))(conv3)
        conv5 = tf.keras.layers.Conv2D(32, (2, 2), (2, 2))(conv4)
        conv6 = tf.keras.layers.Conv2D(64, (2, 2), (2, 2))(conv5)
        
        # Flatten the output so that it matches the shape of the position argument
        conv_flat = tf.keras.layers.Flatten()(conv6)
        conv_out = tf.keras.layers.Dense(1, activation='tanh')(conv_flat)
        
        # Concatenate image output + position
        concatenated = tf.keras.layers.Concatenate()([conv_out, self.position])

        # two outputs as PPO is an actor-critic so we need to both update the actor and the critic part
        layer_out = tf.keras.layers.Dense(num_outputs, activation='tanh')(concatenated)
        self.value_out = tf.keras.layers.Dense(1, name='value_out')(concatenated)
        
        # Assembling everything 
        self.base_model = tf.keras.Model([self.image, self.position], [layer_out, self.value_out])
        self.base_model.summary()
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        """Custom core forward method."""
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(input_dict['obs'], self.obs_space, "tf")

        inputs = {'image': orig_obs["image"], 'position': orig_obs["position"]}
        model_out, self._value_out = self.base_model(inputs)

        return model_out, state
    
    def value_function(self):
        return tf.reshape(self._value_out, [-1])    
