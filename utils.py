import os
import difflib
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from tqdm import tqdm
from scipy.fftpack import fft
# from python_speech_features import mfcc
from random import shuffle

def data_hparams():
	params = tf.contrib.training.HParams(
		# vocab
		data_type='train',
		data_path='dataset/',
		thchs30=False,
		aishell=False,
		prime=False,
		stcmd=False,
		magicdata=True,
		batch_size=1,
		data_length=None,
		shuffle=True)
	return params


class get_data():
	def __init__(self, args):
		self.data_type = args.data_type
		self.data_path = args.data_path
		self.thchs30 = args.thchs30
		self.aishell = args.aishell
		self.prime = args.prime
		self.stcmd = args.stcmd
		self.magicdata = args.magicdata,
		self.data_length = args.data_length
		self.batch_size = args.batch_size
		self.shuffle = args.shuffle
		self.source_init()

	def source_init(self):
		print('get source list...')
		read_files = []
		if self.data_type == 'train':
			if self.thchs30 == True:
				read_files.append('thchs_train.txt')
			if self.aishell == True:
				read_files.append('aishell_train.txt')
			if self.prime == True:
				read_files.append('prime.txt')
			if self.stcmd == True:
				read_files.append('stcmd.txt')
			if self.magicdata:
				read_files.append('train.txt')
		elif self.data_type == 'dev':
			if self.thchs30 == True:
				read_files.append('thchs_dev.txt')
			if self.aishell == True:
				read_files.append('aishell_dev.txt')
			if self.magicdata:
				read_files.append('dev_device.txt')
		elif self.data_type == 'test':
			if self.thchs30 == True:
				read_files.append('thchs_test.txt')
			if self.aishell == True:
				read_files.append('aishell_test.txt')
			if self.magicdata:
				read_files.append('test.txt')
		print(read_files)
		
		self.wav_lst = []
		self.pny_lst = []
		self.han_lst_str = []
		self.han_lst = []
		for file in read_files:
			if "csv" in file:
				pass
			else:
				print('load ', file, ' data...')
				sub_file =  file             # './DeepSpeechRecognition-master/data/'
				with open(sub_file, 'r', encoding='utf8') as f:
					data = f.readlines()
				for line in tqdm(data, ascii=True):
					wav_file, pny, han = line.split('\t')
					self.wav_lst.append(wav_file)
					self.pny_lst.append(pny.split(' '))
					self.han_lst_str.append(han.strip('\n'))
					
		if self.data_length is not None:
			self.wav_lst = self.wav_lst[:self.data_length]
			self.pny_lst = self.pny_lst[:self.data_length]
			self.han_lst_str = self.han_lst_str[:self.data_length]
		print('make am vocab...')
		self.am_vocab = self.mk_am_vocab(self.pny_lst)
		print('make lm pinyin vocab...')
		self.pny_vocab = self.mk_lm_pny_vocab(self.pny_lst)
		print('make lm hanzi vocab...')
		self.han_vocab = self.mk_lm_han_vocab(self.han_lst_str)

	def get_am_batch(self):
		shuffle_list = [i for i in range(len(self.wav_lst))]
		while 1:
			if self.shuffle == True:
				shuffle(shuffle_list)
			for i in range(len(self.wav_lst) // self.batch_size):
				wav_data_lst = []
				label_data_lst = []
				begin = i * self.batch_size
				end = begin + self.batch_size
				sub_list = shuffle_list[begin:end]
				for index in sub_list:
					fbank = compute_fbank(self.wav_lst[index])
                    
					##归一化
					fbank=(fbank-fbank.mean())/fbank.std()
					pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
					pad_fbank[:fbank.shape[0], :] = fbank
					label = self.pny2id(self.pny_lst[index], self.am_vocab)
					label_ctc_len = self.ctc_len(label)
					if pad_fbank.shape[0] // 8 >= label_ctc_len:
						wav_data_lst.append(pad_fbank)
						label_data_lst.append(label)
				pad_wav_data, input_length = self.wav_padding(wav_data_lst)
				pad_label_data, label_length = self.label_padding(label_data_lst)
				inputs = {'the_inputs': pad_wav_data,
						  'the_labels': pad_label_data,
						  'input_length': input_length,
						  'label_length': label_length,
						  }
				outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )}
				yield inputs, outputs

	def get_lm_batch(self):
		batch_num = len(self.pny_lst) // self.batch_size
		for k in range(batch_num):
			begin = k * self.batch_size
			end = begin + self.batch_size
			input_batch = self.pny_lst[begin:end]
			label_batch = self.han_lst[begin:end]
			max_len = max([len(line) for line in input_batch])
			input_batch = np.array(
				[self.pny2id(line, self.pny_vocab) + [0] * (max_len - len(line)) for line in input_batch])
			label_batch = np.array(
				[self.han2id(line, self.han_vocab) + [0] * (max_len - len(line)) for line in label_batch])
			yield input_batch, label_batch

	def pny2id(self, line, vocab):
		return [vocab.index(pny) for pny in line]

	def han2id(self, line, vocab):
		return [vocab.index(han) for han in line]

	def wav_padding(self, wav_data_lst):
		wav_lens = [len(data) for data in wav_data_lst]
		wav_max_len = max(wav_lens)
		wav_lens = np.array([leng // 8 for leng in wav_lens])
		new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
		for i in range(len(wav_data_lst)):
			new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
		return new_wav_data_lst, wav_lens

	def label_padding(self, label_data_lst):
		label_lens = np.array([len(label) for label in label_data_lst])
		max_label_len = max(label_lens)
		new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
		for i in range(len(label_data_lst)):
			new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
		return new_label_data_lst, label_lens

	def mk_am_vocab(self, data):
		vocab = []
		for line in tqdm(data, ascii=True):
			for pny in line:
				if pny not in vocab:
					vocab.append(pny)
		vocab.append('_')
		return vocab

	def mk_lm_pny_vocab(self, data):
		vocab = ['<PAD>']
		for line in tqdm(data, ascii=True):
			for pny in line:
				if pny not in vocab:
					vocab.append(pny)
		return vocab

	def mk_lm_han_vocab(self, data):
		vocab = ['<PAD>']
		# 对每句话进行分析
		for line in tqdm(data, ascii=True):
			_hlst = []
			if line in vocab: # 如果该句话在vocab中
				self.han_lst.append([line])
				continue
			line = line.split(' ') # 分词处理
			for patch in line:
				if patch.encode("utf-8").isalpha(): # 如果全为英文字母
					_hlst.append(patch)
					if patch not in vocab:
						vocab.append(patch)
				elif patch == "[UNK]": # 如果等于特例[UNK]
					_hlst.append(patch)
					print(patch)
				else:               
					en_str = ""
					for han in patch:
						if not is_Chinese(han): # 如果非汉字，就继续循环
							en_str += han
						elif is_Chinese(han):   # 如果是汉字，就终止en_str累加

							if en_str != "":
								_hlst.append(en_str)
							_hlst.append(han)

							if en_str != "" and en_str not in vocab:
								vocab.append(en_str)
							en_str = ""
							if han not in vocab:
								vocab.append(han)
						else:
							print("ERROR:", han)
							
					if en_str != "":
						_hlst.append(en_str)
					if en_str != "" and en_str not in vocab:
						vocab.append(en_str) 
						en_str = ""
			self.han_lst.append(_hlst)
		return vocab

	def ctc_len(self, label):
		add_len = 0
		label_len = len(label)
		for i in range(label_len - 1):
			if label[i] == label[i + 1]:
				add_len += 1
		return label_len + add_len


def is_Chinese(word):
	for ch in word:
		if '\u4e00' <= ch <= '\u9fff':
			return True
	return False

# 对音频文件提取mfcc特征
def compute_mfcc(file):
	fs, audio = wav.read(file)
	mfcc_feat = mfcc(audio, samplerate=fs, numcep=26)
	mfcc_feat = mfcc_feat[::3]
	mfcc_feat = np.transpose(mfcc_feat)
	return mfcc_feat

# 获取信号的时频图
def compute_fbank(file):
	#_path = file.replace(".bin", ".npz")
	#try:
	#	if os.path.exists(_path):
	#		_data = np.load(_path)["data"]
	#		return _data
	#except Exception as e:
	#print("Error:", e)
	x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
	w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
	# fs, wavsignal = wav.read(file)
	fs = 16000
	wavsignal = np.fromfile(file, dtype=np.int16)
	# wav波形 加时间窗以及时移10ms
	time_window = 25  # 单位ms
	wav_arr = wavsignal # np.array(wavsignal)
	range0_end = int(len(wavsignal) / fs * 1000 - time_window) // 10 + 1 # 计算循环终止的位置，也就是最终生成的窗数
	data_input = np.zeros((range0_end, 200), dtype=np.float)  # 用于存放最终的频率特征数据
	data_line = np.zeros((1, 400), dtype=np.float)
	for i in range(0, range0_end):
		p_start = i * 160
		p_end = p_start + 400
		data_line = wav_arr[p_start:p_end]
		data_line = data_line * w  # 加窗
		data_line = np.abs(fft(data_line))
		data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的
	data_input = np.log(data_input + 1)
	# data_input = data_input[::]
	return data_input

# word error rate------------------------------------
def GetEditDistance(str1, str2):
	leven_cost = 0
	s = difflib.SequenceMatcher(None, str1, str2)
	for tag, i1, i2, j1, j2 in s.get_opcodes():
		if tag == 'replace':
			leven_cost += max(i2-i1, j2-j1)
		elif tag == 'insert':
			leven_cost += (j2-j1)
		elif tag == 'delete':
			leven_cost += (i2-i1)
	return leven_cost


# 定义解码器------------------------------------
def decode_ctc(num_result, num2word):
	result = num_result[:, :, :]
	in_len = np.zeros((1), dtype = np.int32)
	in_len[0] = result.shape[1]
	r = K.ctc_decode(result, in_len, greedy = True, beam_width=10, top_paths=1)
	r1 = K.get_value(r[0][0])
	r1 = r1[0]
	text = []
	for i in r1:
		text.append(num2word[i])
	return r1, text

def generate_fbank(x):
	data = compute_fbank(x)
	path = x.replace(".bin", "")
	if not os.path.exists(os.path.dirname(path + ".npz")):
		print("making dir:", os.path.dirname(path))
		os.makedirs(os.path.dirname(path))
	np.savez_compressed(path, data=data)
	return path + ".npz"

def multi_task(data, s_type):
	import swifter
	data["fbank_path"] =  data['wav_file'].swifter.apply(lambda x: generate_fbank("./dataset/dev_byte(Android)/" + os.path.basename(x)))
	data["fbank_path"] =  data['wav_file'].swifter.apply(lambda x: generate_fbank("./dataset/dev_byte(IOS)/" + os.path.basename(x)))
	data["fbank_path"] =  data['wav_file'].swifter.apply(lambda x: generate_fbank("./dataset/dev_byte(Recorder)/" + os.path.basename(x)))
	return data
	
if __name__ == "__main__":
	data_args = data_hparams()
	data_args.data_type = 'train'
	data_args.data_path = './dataset/'
	# data_args.thchs30 = True
	# data_args.aishell = True
	# data_args.prime = True
	# data_args.stcmd = True
	data_args.magicdata = True
	data_args.batch_size = 4
	# data_args.data_length = None
	data_args.shuffle = True
	train_data = get_data(data_args)
	print(train_data.get_am_batch())