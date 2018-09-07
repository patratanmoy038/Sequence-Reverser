import time
import numpy as np

import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.layers import embed_sequence
from tensorflow.contrib.seq2seq import tile_batch, BahdanauAttention, AttentionWrapper, TrainingHelper, BasicDecoder, BeamSearchDecoder, dynamic_decode, sequence_loss
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops


from Model import Model


class Config(object):
    train = True
    n_samples = 100000
    n_val_samples = 1000
    vocab_size = 10
    num_units = 64
    embedding_size = 5
    batch_size = n_val_samples
    max_sequence_length = 10
    beam_width = 3
    lr = 0.01
    n_epochs = 20


class DataLoader(object):
    def __init__(self, config):
        self.config = config
        np.random.seed(0)

        self.x_val = np.full((self.config.n_val_samples, self.config.max_sequence_length + 1), self.config.vocab_size, dtype=np.int32)
        self.y_val = np.full((self.config.n_val_samples, self.config.max_sequence_length + 2), self.config.vocab_size, dtype=np.int32)
        self.s_val = np.zeros((self.config.n_val_samples,), dtype=np.int32)
        for i in range(self.config.n_val_samples):
            seq_len = np.random.randint(1, self.config.max_sequence_length + 1)
            input_seq = np.random.randint(0, self.config.vocab_size, seq_len)

            self.x_val[i, :seq_len] = input_seq
            self.y_val[i, :seq_len + 1] = np.append(np.array([self.config.vocab_size + 1]), input_seq[::-1])
            self.s_val[i] = seq_len + 1

    def get_minibatches(self):
        # index - 0:vocab -> True Vocabulary
        # index - vocab -> END
        # index - vocab + 1 -> START
        x = np.full((self.config.batch_size, self.config.max_sequence_length + 1), self.config.vocab_size, dtype=np.int32)
        y = np.full((self.config.batch_size, self.config.max_sequence_length + 2), self.config.vocab_size, dtype=np.int32)
        s = np.zeros((self.config.batch_size,), dtype=np.int32)
        for i in range(self.config.batch_size):
            seq_len = np.random.randint(1, self.config.max_sequence_length + 1)
            input_seq = np.random.randint(0, self.config.vocab_size, seq_len)

            x[i, :seq_len] = input_seq
            y[i, :seq_len + 1] = np.append(np.array([self.config.vocab_size + 1]), input_seq[::-1])
            s[i] = seq_len + 1

        return x, s, y


class SoftmaxModel(Model):
    def add_placeholders(self):
        self.inputs = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_sequence_length + 1], name='sequence')
        self.labels = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_sequence_length + 2])
        self.lengths = tf.placeholder(tf.int32, [self.config.batch_size, ], name='sequence_length')

    def create_feed_dict(self, inputs_batch, length_batch, labels_batch=None):
        feed_dict = {
            self.inputs: inputs_batch,
            self.lengths: length_batch,
        }
        if labels_batch is not None:
            feed_dict[self.labels] = labels_batch

        return feed_dict

    def add_prediction_op(self):
        encoder_embed_seq = embed_sequence(
                                self.inputs,
                                vocab_size=self.config.vocab_size + 2,
                                embed_dim=self.config.embedding_size,
                                scope='embed'
                            )

        decoder_input_embed_seq = embed_sequence(
                                    self.labels[:, :-1],
                                    vocab_size=self.config.vocab_size + 2,
                                    embed_dim=self.config.embedding_size,
                                    scope='embed',
                                    reuse=True
                                )

        with tf.variable_scope('embed', reuse=True):
            embeddings = tf.get_variable('embeddings')

        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                                BasicLSTMCell(self.config.num_units, name="encoder"),
                                encoder_embed_seq,
                                dtype=tf.float32,
                                sequence_length=self.lengths,
                            )

        if self.config.train:
            tiled_encoder_outputs = encoder_outputs
            tiled_encoder_final_state = encoder_final_state
            tiled_sequence_length = self.lengths
        else:
            tiled_encoder_outputs = tile_batch(
                                        encoder_outputs, multiplier=self.config.beam_width)
            tiled_encoder_final_state = tile_batch(
                                        encoder_final_state, multiplier=self.config.beam_width)
            tiled_sequence_length = tile_batch(
                                        self.lengths, multiplier=self.config.beam_width)

        attention_mechanism = BahdanauAttention(
                                num_units=self.config.num_units,
                                memory=tiled_encoder_outputs,
                                memory_sequence_length=tiled_sequence_length)

        attn_cell = AttentionWrapper(
                        BasicLSTMCell(self.config.num_units, name="decoder"),
                        attention_mechanism,
                        attention_layer_size=self.config.num_units / 2)

        if self.config.train:
            batch_size = self.config.batch_size
        else:
            batch_size = self.config.batch_size * self.config.beam_width

        decoder_initial_state = attn_cell.zero_state(
                                    dtype=tf.float32,
                                    batch_size=batch_size)
        decoder_initial_state = decoder_initial_state.clone(
                                    cell_state=tiled_encoder_final_state)

        output_layer = tf.layers.Dense(self.config.vocab_size + 2, use_bias=True, name='output_projection')

        if self.config.train:
            training_helper = TrainingHelper(inputs=decoder_input_embed_seq,
                                                sequence_length=self.lengths,
                                                name='training_helper')

            decoder = BasicDecoder(
                                cell=attn_cell,
                                helper=training_helper,
                                initial_state=decoder_initial_state,
                                output_layer=output_layer)
        else:
            def embed_and_input_proj(inputs):
                return tf.nn.embedding_lookup(embeddings, inputs)

            start_tokens = tf.ones([self.config.batch_size, ], tf.int32) * (self.config.vocab_size + 1)
            decoder = BeamSearchDecoder(
                                cell=attn_cell,
                                embedding=embed_and_input_proj,
                                start_tokens=start_tokens,
                                end_token=self.config.vocab_size,
                                initial_state=decoder_initial_state,
                                beam_width=self.config.beam_width,
                                output_layer=output_layer,)

        if self.config.train:
            decoder_outputs, _, _ = dynamic_decode(
                                        decoder=decoder,
                                        impute_finished=True,
                                        maximum_iterations=self.config.max_sequence_length + 1
                                        )
            pred_logits = tf.identity(decoder_outputs.rnn_output, name="prediction")
        else:
            decoder_outputs, _, _ = dynamic_decode(
                                        decoder=decoder,
                                        impute_finished=False,
                                        maximum_iterations=self.config.max_sequence_length + 1
                                        )
            pred_logits = tf.identity(decoder_outputs.predicted_ids, name="prediction")
        return pred_logits

    def add_loss_op(self, pred):
        masks = tf.sequence_mask(
                    lengths=self.lengths,
                    dtype=tf.float32,
                    maxlen=self.config.max_sequence_length + 1,
                    name='masks')

        loss = sequence_loss(
                    logits=pred,
                    targets=self.labels[:, 1:],
                    weights=masks,
                    average_across_timesteps=True,
                    average_across_batch=True,)

        return loss

    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op

    def validate(self, sess):
        x = self.data.x_val
        s = self.data.s_val
        y = self.data.y_val

        i = np.random.randint(x.shape[0])

        out = np.argmax(self.predict_on_batch(sess, x, s, y), axis=-1)

        acc = 0.
        for j, (pred, true) in enumerate(zip(out, y)):
            EOS = np.argmax(pred)
            if j == i:
                print "Random Valid Sample - Input:", x[i, :s[i]-1], "True:", true[1:s[i]], "Pred:", pred[:EOS], "\n"
            if (pred[:EOS] == true[1:EOS+1]).all():
                acc += 1.
        acc = acc / self.config.n_val_samples
        return acc

    def test(self, sess):
        x, s, y = self.data.get_minibatches()
        pred_logits = self.predict_on_batch(sess, x, s)[:, :, 0]

        acc = 0.
        i = np.random.randint(x.shape[0])
        for j, (pred, true) in enumerate(zip(pred_logits, y)):
            EOS = np.argmax(pred)
            if j == i:
                print "Random Test Sample - Input:", x[i, :s[i]-1], "True:", true[1:s[i]], "Pred:", pred[:EOS], "\n"
            if (pred[:EOS] == true[1:EOS+1]).all():
                acc += 1.
        acc = acc / self.config.batch_size
        print "Test Accuracy using BeamSearch", acc

    def run_epoch(self, sess):
        test_acc = self.validate(sess)
        total_loss = 0
        for n_minibatches in range(self.config.n_samples // self.config.batch_size):
            input_batch, length_batch, labels_batch = self.data.get_minibatches()
            total_loss += self.train_on_batch(sess, input_batch, length_batch, labels_batch)
        return total_loss / (n_minibatches + 1), test_acc

    def load_network(self, sess):
        checkpoint = tf.train.get_checkpoint_state('saved/')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(sess, checkpoint.model_checkpoint_path)
            print 'Successfully Loaded:', checkpoint.model_checkpoint_path

    def fit(self, sess):
        best_acc = 0.
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            average_loss, test_acc = self.run_epoch(sess)
            duration = time.time() - start_time
            print 'Epoch {:}: Training Loss = {:.8f} | Validation Accuracy = {:.4f} ({:.3f} sec)'.format(epoch, average_loss, test_acc, duration)
            if test_acc >= best_acc:
                saved_path = self.saver.save(sess, "saved/model.ckpt", global_step=epoch)
                print 'Successfully Saved:', saved_path
                best_acc = test_acc
        print "Best Validation Accuracy:", best_acc

    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.build()
        self.saver = tf.train.Saver()


def freeze_graph(sess):
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        ["prediction"]
        )

    with tf.gfile.GFile('saved/deploy/frozen_model.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


def serialize_graph(sess):
    builder = tf.saved_model.builder.SavedModelBuilder("SavedModel")
    builder.add_meta_graph_and_variables(sess, ["prediction"])
    builder.save()


if __name__ == "__main__":
    config = Config()
    data = DataLoader(config)
    model = SoftmaxModel(config, data)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        model.load_network(session)

        # for i, n in enumerate(tf.get_default_graph().as_graph_def().node):
        #     print i, n.name
        model.fit(session)
        # model.test(session)
        serialize_graph(session)
        session.close()
