import tensorflow as tf
import keras.backend as K

rf = tf.contrib.receptive_field.compute_receptive_field_from_graph_def

def get_rf_info(graph_def, input_name, output_name):
	rf_x, rf_y, eff_stride_x, eff_stride_y, eff_pad_x, eff_pad_y = rf(graph_def, input_name, output_name)
	return([rf_x, rf_y, eff_stride_x, eff_stride_y, eff_pad_x, eff_pad_y])

def get_all_rf(model):
	graph = K.get_session().graph
	graph_def = graph.as_graph_def()
	out = []
	input_name = model.layers[0].name
	for layer in model.layers:
		if "conv"in layer.name.lower():
			info = get_rf_info(graph_def, input_name, layer.name + "/convolution")
			out.append([layer.name] + info)
		elif "pool" in layer.name.lower():
			info = get_rf_info(graph_def, input_name, layer.name + "/MaxPool")
			out.append([layer.name] + info)
	return out