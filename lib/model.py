import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.gru = tf.keras.layers.GRU(hidden_size,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.hidden_size = hidden_size

    def call(self, x):
        x = self.embedding(x)
        state = self.initialize_hidden_state(x)
        output, state = self.gru(x, initial_state=state)
        return output, state

    def initialize_hidden_state(self, x):
        return tf.zeros((x.shape[0], self.hidden_size))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.gru = tf.keras.layers.GRU(hidden_size,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.projection = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden):
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # passing the concatenated vector to the GRU
        output, state = self.gru(x, hidden)

        # output shape == (batch_size, vocab)
        x = self.projection(state)

        return x, state


class Seq2Seq(tf.keras.Model):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 embedding_size,
                 hidden_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(src_vocab_size, embedding_size, hidden_size)
        self.decoder = Decoder(tgt_vocab_size, embedding_size, hidden_size)
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

    def call(self, x, t):
        _, state = self.encoder(x)
        loss = 0.
        for y, o in zip(t[:-1], t[1:]):
            p, state = self.decoder(y, state)
            print(p.shape, o.shape)
            loss += self.loss_func(p, o)
        return loss
