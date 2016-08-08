import tensorflow as tf
import numpy as np

from utils import util

from SEP_EP_tensorflow.Fa import Fa

def main(n_iter = 100, learning_rate = 1e-1, num_samples = 1):

	#config = tf.ConfigProto(
    #    device_count = {'GPU': 0}
    #)

	#gpu_opt=tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
	#config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_opt)
	#sess = tf.Session(config=config)


	sess = tf.Session()

	data_size = 1000
	model = Fa(data_size, dimx =3, dimz = 2)
	model.fit(sess, n_iter = n_iter, learning_rate = learning_rate)










if __name__ == "__main__":
	main()