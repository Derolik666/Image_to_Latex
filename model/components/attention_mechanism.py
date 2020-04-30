import tensorflow as tf


class AttentionMechanism(object):
    """Class to compute attention over an image"""

    def __init__(self, img, dim_e, tiles=1):
        """Stores the image under the right shape.

        We loose the H, W dimensions and merge them into a single
        dimension that corresponds to "regions" of the image.

        Args:
            img: (tf.Tensor) image
            dim_e: (int) dimension of the intermediary vector used to
                compute attention
            tiles: (int) default 1, input to context h may have size
                    (tile * batch_size, ...)

        """
        if len(img.shape) == 3:
            self._img = img
        elif len(img.shape) == 4:
            N = tf.shape(img)[0]
            H, W = tf.shape(img)[1], tf.shape(img)[2]  # image
            C = img.shape[3].value                 # channels
            self._img = tf.reshape(img, shape=[N, H*W, C])
        else:
            print("Image shape not supported")
            raise NotImplementedError

        # dimensions
        self._n_regions = tf.shape(self._img)[1]
        self._n_channels = self._img.shape[2].value
        self._dim_e = dim_e
        self._tiles = tiles
        self._scope_name = "att_mechanism"
        # img [batch_size, H*W, 1024]  [512, 512] att_img [batch_size, H*W, 512]
        # attention vector over the image
        self._att_img = tf.layers.dense(
            inputs=self._img,
            units=self._dim_e,
            use_bias=False,
            name="att_img")

    def context(self, h, s):
        """Computes attention

        Args:
            h: (batch_size, num_units) hidden state

        Returns:
            c: (batch_size, channels) context vector

        """
        with tf.variable_scope(self._scope_name):
            if self._tiles > 1:
                att_img = tf.expand_dims(self._att_img, axis=1)
                att_img = tf.tile(att_img, multiples=[1, self._tiles, 1, 1])
                att_img = tf.reshape(att_img, shape=[-1, self._n_regions,
                                                     self._dim_e])
                img = tf.expand_dims(self._img, axis=1)
                img = tf.tile(img, multiples=[1, self._tiles, 1, 1])
                img = tf.reshape(img, shape=[-1, self._n_regions,
                                             self._n_channels])
            else:
                att_img = self._att_img
                img = self._img
            # s [batch_size, 1024] att_s [batch_size, 512]
            att_s = tf.layers.dense(
                inputs=s, units=self._dim_e, use_bias=False)
            att_s = tf.expand_dims(att_s, axis=1)
            # att_img [batch_size, H*W+1, 512]
            att_concat = tf.concat([att_img, att_s], axis=1)

            # h [batch_size, 512] [512, 512] [batch_size, 512]
            # computes attention over the hidden vector
            att_h = tf.layers.dense(
                inputs=h, units=self._dim_e, use_bias=False)

            # sums the two contributions
            att_h = tf.expand_dims(att_h, axis=1)
            # att [batch_size, H*W+1, 512]
            att = tf.tanh(att_concat + att_h)

            # computes scalar product with beta vector
            # works faster with a matmul than with a * and a tf.reduce_sum
            att_beta = tf.get_variable("att_beta", shape=[self._dim_e, 1],
                                       dtype=tf.float32)
            # att_flat [batch*(H*W+1), 512]
            att_flat = tf.reshape(att, shape=[-1, self._dim_e])
            # z_hat [batch*(H*W+1), 1]
            z_hat = tf.matmul(att_flat, att_beta)
            # z_hat [batch, H*W+1]
            z_hat = tf.reshape(z_hat, shape=[-1, self._n_regions + 1])
            # alpha_split1 [batch, H*W] alpha_split2 [batch, 1]
            alpha_split1, alpha_split2 = tf.split(
                z_hat, [self._n_regions, 1], axis=1)

            # compute weights
            alpha = tf.nn.softmax(alpha_split1)
            # alpha [batch, H*W, 1]
            alpha = tf.expand_dims(alpha, axis=-1)
            # c [batch, 1024]
            c = tf.reduce_sum(alpha * img, axis=1)

            # compute weights
            alpha_hat = tf.nn.softmax(z_hat)
            # beta_split1 [batch, H*W] beta_split2 [batch, 1]
            beta_split1, beta_split2 = tf.split(
                alpha_hat, [self._n_regions, 1], axis=1)
            # beta_split2 [batch, 1]  s [batch_size, 1024] beta_s [batch_size, 1024]
            beta_s = beta_split2 * s
            # one_vector [batch, 1]
            one_vector = tf.ones([tf.shape(att_img)[0], 1], tf.float32)
            # oppo_beta [batch, 1]
            oppo_beta = one_vector - beta_split2
            # oppo_beta [batch, 1] c [batch, 1024] oppo_beta_c [batch, 1024]
            oppo_beta_c = oppo_beta * c
            # beta_s [batch_size, 1024] oppo_beta_c [batch, 1024] c_hat[batch_size, 1024]
            c_hat = beta_s + oppo_beta_c

            return c_hat

    def initial_cell_state(self, cell):
        """Returns initial state of a cell computed from the image

        Assumes cell.state_type is an instance of named_tuple.
        Ex: LSTMStateTuple

        Args:
            cell: (instance of RNNCell) must define _state_size

        """
        _states_0 = []
        for hidden_name in cell._state_size._fields:
            hidden_dim = getattr(cell._state_size, hidden_name)
            h = self.initial_state(hidden_name, hidden_dim)
            _states_0.append(h)

        initial_state_cell = type(cell.state_size)(*_states_0)

        return initial_state_cell

    def initial_state(self, name, dim):
        """Returns initial state of dimension specified by dim"""
        with tf.variable_scope(self._scope_name):
            img_mean = tf.reduce_mean(self._img, axis=1)
            W = tf.get_variable("W_{}_0".format(name), shape=[self._n_channels,
                                                              dim])
            b = tf.get_variable("b_{}_0".format(name), shape=[dim])
            h = tf.tanh(tf.matmul(img_mean, W) + b)

            return h
