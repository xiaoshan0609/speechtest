import os
#sys.path.append(r'/kaggle/working/speechtest')
import tensorflow as tf
from utils import get_data, data_hparams
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,TensorBoard



import keras.backend.tensorflow_backend as ktf
# 指定GPUID, 第一块GPU可用
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
# # GPU 显存自动分配
# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allow_growth=True
# #config.gpu_options.per_process_gpu_memory_fraction = 0.3
# session = tf.Session(config=config)
# ktf.set_session(session)

# class MyCbk(keras.callbacks.Callback):
	# def __init__(self, model):
		# self.model_to_save = model
	# def on_epoch_end(self, epoch, logs = None):
		# ckpt = "model_{epoch:02d}-{val_loss:.2f}.hdf5"
		# self.model_to_save.save(os.path.join('./checkpoint', ckpt))
		
class MyCbk(ModelCheckpoint):
	def __init__(self, model, filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1):
		self.single_model = model
		super(MyCbk, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)
	def set_model(self, model):
		super(MyCbk, self).set_model(self.single_model)

		
# 0.准备训练所需数据------------------------------
data_args = data_hparams()
data_args.data_type = 'train'
data_args.data_path = './dataset/'
# data_args.thchs30 = True
# data_args.aishell = True
# data_args.prime = True
# data_args.stcmd = True
data_args.magicdata = True
data_args.batch_size = 16
# data_args.data_length = None
data_args.shuffle = True
train_data = get_data(data_args)

# 0.准备验证所需数据------------------------------
data_args = data_hparams()
data_args.data_type = 'dev'
data_args.data_path = './dataset/'
data_args.thchs30 = False
data_args.aishell = False
data_args.prime = False
data_args.stcmd = False
data_args.batch_size = 16
# data_args.data_length = None
data_args.shuffle = True
dev_data = get_data(data_args)

# 1.声学模型训练-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams
am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am_args.gpu_nums = 0
am_args.lr = 1e-4
am_args.is_training = True
am = Am(am_args)

# if os.path.exists('logs_am/model.h5'):
	# print('load acoustic model...')
	# am.ctc_model.load_weights('checkpoint/model_04-86.93.hdf5')

epochs = 100
batch_num = len(train_data.wav_lst) // train_data.batch_size
print("len(train_data.wav_lst):", len(train_data.wav_lst))
print("batch_num:", batch_num)
batch_num_val = len(dev_data.wav_lst) // dev_data.batch_size
print("batch_num_val:", batch_num_val)

# checkpoint

#ckpt = "model_{epoch:02d}-{val_loss:.2f}.hdf5"
#ckpt = "model_{epoch:02d}-{val_loss:.2f}.hdf5"
#checkpoint = ModelCheckpoint(os.path.join('F:/speechtest/算法与数据/code0106/checkpoint', ckpt), monitor='val_loss', save_weights_only=False, verbose=1, save_best_only=True)

#checkpoint = ModelCheckpoint(os.path.join('./checkpoint', ckpt), monitor='val_loss', save_weights_only=False, verbose=1, save_best_only=True)
#checkpoint = MyCbk(am.ctc_model, os.path.join('./checkpoint', ckpt), monitor='val_loss', save_weights_only=False, verbose=1, save_best_only=True)
#earlystopping = EarlyStopping(monitor='val_loss', verbose=1, patience=3, restore_best_weights=True)
#reducelronplateau = ReduceLROnPlateau(monitor="val_loss", verbose=1, mode='min', factor=0.1, patience=1)
#
# for k in range(epochs):
#     print('this is the', k+1, 'th epochs trainning !!!')
#     batch = train_data.get_am_batch()
#     dev_batch = dev_data.get_am_batch()
#     am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=10, callbacks=[checkpoint], workers=1, use_multiprocessing=False, validation_data=dev_batch, validation_steps=200)

batch = train_data.get_am_batch()
dev_batch = dev_data.get_am_batch()

# tensorborad查看整个模型训练过程
tbCallBack = TensorBoard(log_dir="./logs_am/model")
LOG_DIR = './logs_am/model'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)
get_ipython().system_raw('./ngrok http 6006 &')
!curl -s http://localhost:4040/api/tunnels | python3 -c \"import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"



if am_args.gpu_nums <= 1:
	am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=epochs, callbacks=[tbCallBack], workers=1, use_multiprocessing=False, validation_data=dev_batch, validation_steps=batch_num_val)
	# 这个带上面就报错
	#am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=epochs,  workers=1,use_multiprocessing=False )

else:
	am.parallel_ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=epochs, callbacks=[earlystopping, reducelronplateau, checkpoint], workers=1, use_multiprocessing=False, validation_data=dev_batch, validation_steps=batch_num_val)

am.ctc_model.save_weights('./logs_am/model.h5')


# 2.语言模型训练-------------------------------------------
from model_language.transformer import Lm, lm_hparams
lm_args = lm_hparams()
lm_args.num_heads = 8
lm_args.num_blocks = 6
lm_args.input_vocab_size = len(train_data.pny_vocab)
lm_args.label_vocab_size = len(train_data.han_vocab)
lm_args.max_length = 100
lm_args.hidden_units = 512
lm_args.dropout_rate = 0.33
lm_args.lr = 0.0001
lm_args.is_training = True
lm = Lm(lm_args)

epochs = 100
with lm.graph.as_default():
	saver =tf.train.Saver()
with tf.Session(graph=lm.graph) as sess:
	merged = tf.summary.merge_all()
	sess.run(tf.global_variables_initializer())
	# add_num = 0
	# if os.path.exists('logs_lm/checkpoint'):
	# 	print('loading language model...')
	# 	latest = tf.train.latest_checkpoint('logs_lm')
	# 	add_num = int(latest.split('_')[-1])
	# 	saver.restore(sess, latest)
	writer = tf.summary.FileWriter('./logs_lm/tensorboard', tf.get_default_graph())
    LOG_DIR = './logs_lm/tensorboard'
    get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
    )
    get_ipython().system_raw('./ngrok http 6006 &')
    !curl -s http://localhost:4040/api/tunnels | python3 -c \"import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
	for k in range(epochs):
		total_loss = 0
		batch = train_data.get_lm_batch()
		for i in range(batch_num):
			input_batch, label_batch = next(batch)
			feed = {lm.x: input_batch, lm.y: label_batch}
			cost,_ = sess.run([lm.mean_loss, lm.train_op], feed_dict=feed)
			total_loss += cost
			if (k * batch_num + i) % 10 == 0:
				rs=sess.run(merged, feed_dict=feed)
				writer.add_summary(rs, k * batch_num + i)
		print('epochs', k+1, ': average loss = ', total_loss/batch_num)
	saver.save(sess, './logs_lm/model.hl') #保存最终的模型
	writer.close()
