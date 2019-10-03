from argparse import ArgumentParser

import tensorflow as tf
import lib


parser = ArgumentParser()
parser.add_argument('--src_path', type=str, default='dataset/train.en')
parser.add_argument('--tgt_path', type=str, default='dataset/train.ja')
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--batch', type=int, default=64)
args = parser.parse_args()

d, src_vocab_size, tgt_vocab_size = lib.load_dataset(args.src_path, args.tgt_path)
d = d.padded_batch(args.batch, (100, 100)).shuffle(args.batch * 100)
opt = tf.keras.optimizers.Adam()
model = lib.Seq2Seq(src_vocab_size, tgt_vocab_size, 100, 100)
trainer = lib.Trainer(args.epoch, model, d, opt)
trainer.run()
