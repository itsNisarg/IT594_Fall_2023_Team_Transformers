import tensorflow as tf
from tensorflow.python.training import training_util
from training_helper import train_model
from predict_helper import predict
from batcher import batcher
from transformer import Transformer
import os
from data_helper import Vocab



def my_model(features, labels, mode, params):

	transformer = Transformer(num_layers=params["num_layers"], d_model=params["model_depth"], num_heads=params["num_heads"], dff=params["dff"], vocab_size=params["vocab_size"], batch_size=params["batch_size"])
	
	checkpoint_dir = "{}/checkpoint".format(params["model_dir"])
	ckpt = tf.train.Checkpoint(transformer=transformer)
	
	# restore from checkpoint
	ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir))

	predictions, attn_weights = predict(features, params, transformer)
	estimator_spec = tf.estimator.EstimatorSpec(mode,  predictions={"predictions":predictions})
  
	print(transformer.summary())
	return estimator_spec


def build_model(params):

	config = tf.estimator.RunConfig(
		tf_random_seed=params["seed"], 
		log_step_count_steps=params["log_step_count_steps"],
		save_summary_steps=params["save_summary_steps"]
	)

	return tf.estimator.Estimator(
					model_fn=my_model,
					params=params, config=config, model_dir=params["model_dir"] )


def train(params):
	assert params["training"], "change training mode to true"

	tf.compat.v1.logging.info("Building the model ...")
	transformer = Transformer(
		num_layers=params["num_layers"], d_model=params["model_depth"], num_heads=params["num_heads"], dff=params["dff"], 
		vocab_size=params["vocab_size"], batch_size=params["batch_size"])


	tf.compat.v1.logging.info("Creating the batcher ...")
	b = batcher(params["data_dir"], params["vocab_path"], params)

	tf.compat.v1.logging.info("Creating the checkpoint manager")
	logdir = "{}/logdir".format(params["model_dir"])
	checkpoint_dir = "{}/checkpoint".format(params["model_dir"])
	ckpt = tf.train.Checkpoint(step=tf.Variable(0), transformer=transformer)
	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=11)

	ckpt.restore(ckpt_manager.latest_checkpoint)
	if ckpt_manager.latest_checkpoint:
		print("Restored from {}".format(ckpt_manager.latest_checkpoint))
	else:
		print("Initializing from scratch.")

	tf.compat.v1.logging.info("Starting the training ...")
	train_model(transformer, b, params, ckpt, ckpt_manager)

def eval(model, params):
	pass


def test(model, params):
	assert not params["training"], "change training mode to false"
	checkpoint_dir = "{}/checkpoint".format(params["model_dir"])
	logdir = "{}/logdir".format(params["model_dir"])

	pred = model.predict(input_fn = lambda :  batcher(params["data_dir"], params["vocab_path"], params), 
		yield_single_examples=False)
	# print("------------------")
	# print(type(pred))
	# print("------------------")
	yield next(pred)
	# print("------------------")
	# print(type(prediction))
	# print("------------------")
	# print(prediction)

	# print(prediction['predictions'])
	# print(prediction['predictions'].shape)
	# print(prediction['predictions'][0])
	# print(prediction['predictions'][0].shape)

	# vocab = Vocab('tfrecords_finished_files/vocab', 50000)

	# # map the prediction ids to words from vocab
	# prediction_words = []
	# for i in prediction['predictions'][0]:
	# 	prediction_words.append(vocab.id_to_word(i))
	
	# print(prediction_words)

	# yield prediction




# predict the output of the model and write it to file system in correct format 
# def test(model, params):
# 	assert not params["training"], "change training mode to false"
# 	checkpoint_dir = "{}/checkpoint".format(params["model_dir"])
# 	logdir = "{}/logdir".format(params["model_dir"])

# 	pred = model.predict(input_fn = lambda :  batcher(params["data_dir"], params["vocab_path"], params), 
# 		yield_single_examples=False)
	
# 	for p in pred:
# 		article = p["article"].decode()
# 		abstract = p["abstract"].decode()
# 		prediction = p["predictions"].decode()
# 		with open(os.path.join(params["data_dir"], "predictions.txt"), "a") as f:
# 			f.write("Article: {}\nAbstract: {}\nPrediction: {}\n\n".format(article, abstract, prediction))

