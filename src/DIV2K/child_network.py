from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf

from src.DIV2K.models import Model
from src.DIV2K.image_ops import conv
from src.DIV2K.image_ops import fully_connected
from src.DIV2K.image_ops import batch_norm
from src.DIV2K.image_ops import batch_norm_with_mask
from src.DIV2K.image_ops import relu
from src.DIV2K.image_ops import max_pool
from src.DIV2K.image_ops import drop_path
from src.DIV2K.image_ops import global_avg_pool

from src.utils import count_model_params
from src.utils import get_train_ops
from src.ops_general import create_weight
from src.ops_general import create_bias


class ChildNetwork(Model):
    def __init__(self,
                 images,
                 labels,
                 meta_data,
                 output_dir = "./output",
                 use_aux_heads=False,
                 use_model=None,
                 fine_tune=False,
                 feature_fusion=False,
                 channel_attn = False,
                 cutout_size=None,
                 fixed_arc=None,
                 upsample_size=2,
                 num_layers=2,
                 num_cells=5,
                 out_filters=24,
                 sfe_filters=64,
                 # keep_prob=1.0,
                 # drop_path_keep_prob=None,
                 batch_size=32,
                 eval_batch_size=100,
                 test_batch_size=1,
                 clip_mode=None,
                 grad_bound=None,
                 l2_reg=1e-4,
                 lr_init=1e-4,
                 lr_dec_start=0,
                 lr_warmup_val = None,
                 lr_warmup_steps = 5,
                 lr_dec_every=10000,
                 lr_dec_rate=0.1,
                 lr_dec_min=1e-5,
                 lr_cosine=False,
                 lr_max=None,
                 lr_min=None,
                 lr_T_0=None,
                 lr_T_mul=None,
                 num_epochs=None,
                 it_per_epoch=1000,
                 optim_algo=None,
                 sync_replicas=False,
                 num_branches=3,
                 num_aggregate=None,
                 num_replicas=None,
                 data_format="NHWC",
                 name="child",
                 **kwargs
                 ):
        """
        """

        super(self.__class__, self).__init__(
            images,
            labels,
            meta_data,
            output_dir=output_dir,
            cutout_size=cutout_size,
            use_model=use_model,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            test_batch_size=test_batch_size,
            clip_mode=clip_mode,
            grad_bound=grad_bound,
            l2_reg=l2_reg,
            lr_init=lr_init,
            it_per_epoch=it_per_epoch,
            lr_dec_start=lr_dec_start,
            lr_warmup_val = lr_warmup_val,
            lr_warmup_steps = lr_warmup_steps,
            lr_dec_every=lr_dec_every,
            lr_dec_rate=lr_dec_rate,
            lr_dec_min=lr_dec_min,
            # keep_prob=keep_prob,
            optim_algo=optim_algo,
            sync_replicas=sync_replicas,
            num_aggregate=num_aggregate,
            num_replicas=num_replicas,
            data_format=data_format,
            name=name)

        if self.data_format == "NHWC":
            self.actual_data_format = "channels_last"
        elif self.data_format == "NCHW":
            self.actual_data_format = "channels_first"
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

        self.use_model = use_model
        self.fine_tune = fine_tune
        self.use_aux_heads = use_aux_heads
        self.num_epochs = num_epochs
        self.num_train_steps = self.num_epochs * self.num_train_batches
        self.feature_fusion = feature_fusion
        self.channel_attn = channel_attn
        # self.drop_path_keep_prob = drop_path_keep_prob
        self.lr_cosine = lr_cosine
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_T_0 = lr_T_0
        self.lr_T_mul = lr_T_mul
        self.upsample_size = upsample_size
        self.out_filters = out_filters
        self.sfe_filters = sfe_filters
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.num_branches = num_branches
        # self.now_arc = tf.TensorArray(tf.int32,size=0,dynamic_size=True)

        self.fixed_arc = fixed_arc
        if fixed_arc is not None:
            self.exist_fixed_arc = True
        else:
            self.exist_fixed_arc = False
        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="global_step")

        # if self.drop_path_keep_prob is not None:
        #     assert num_epochs is not None, "Need num_epochs to drop_path"

        if self.use_aux_heads:
            self.aux_head_indices = self.num_layers // 2
        
    def _get_C(self, x):
        """
        Args:
          x: tensor of shape [N, H, W, C] or [N, C, H, W]
        """
        if self.data_format == "NHWC":
            return x.get_shape()[3].value
        elif self.data_format == "NCHW":
            return x.get_shape()[1].value
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

    def _get_HW(self, x):
        """
        Args:
          x: tensor of shape [N, H, W, C] or [N, C, H, W]
        """
        return x.get_shape()[2].value

    def _get_strides(self, stride):
        """
        Args:
          x: tensor of shape [N, H, W, C] or [N, C, H, W]
        """
        if self.data_format == "NHWC":
            return [1, stride, stride, 1]
        elif self.data_format == "NCHW":
            return [1, 1, stride, stride]
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

    def _model(self, images, is_training, reuse=False):
        """Compute the predictions given the images."""
        # self.now_arc = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        with tf.variable_scope(self.name, reuse=reuse):
            # the first two inputs
            with tf.variable_scope("stem_conv"):
                # w = create_weight("w", [3, 3, 3, self.out_filters * 3])
                w = create_weight("w_grl", [3, 3, 3, self.sfe_filters])
                b = create_bias("b_grl", [self.sfe_filters])
                x_grl = tf.nn.conv2d(
                    images, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                print("Layer x_grl: {}".format(x_grl))
                w = create_weight("w_sfe", [3, 3, self.sfe_filters, self.out_filters])
                b = create_weight("b_sfe", [self.out_filters])
                x_sfe = tf.nn.conv2d(
                    x_grl, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                # x_sfe = tf.Print(x_sfe, [1], message="x_sfe : ")
                print("Layer x_sfe: {}".format(x_sfe))
                

            if self.data_format == "NHWC":
                split_axis = 3
            elif self.data_format == "NCHW":
                split_axis = 1
            else:
                raise ValueError("Unknown data_format '{0}'".format(self.data_format))
            layers = [x_sfe]
            cell_outputs = []
            cell_outputs.append(x_sfe)
            # building layers in the micro space
            out_filters = self.out_filters
            for layer_id in range(self.num_layers):
                with tf.variable_scope("layer_{0}".format(layer_id)):
                    if self.exist_fixed_arc:
                        x = self._fixed_block(layers, self.fixed_arc, out_filters, 1, is_training)
                    else:
                        x = self._dnas_block(layers, self.fixed_arc, out_filters)
                    # x = tf.Print(x, [1], message="x : ")
                    print("Layer {0:>2d}: {1}".format(layer_id, x))
                    cell_outputs.append(x)
                    with tf.variable_scope("block_connection_{}".format(layer_id)):
                        next_inp = tf.concat(cell_outputs, 3)
                        w = create_weight("w_bc",[1,1, out_filters*len(cell_outputs), out_filters])
                        b = create_bias("b_bc", [out_filters])
                        layers = [tf.nn.conv2d(
                            next_inp, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b]

            # print("Layers in cell_outputs: {}".format(cell_outputs))
            if self.feature_fusion:
                gff_arc=self.fixed_arc[-self.num_layers:]
                print("feature_fusion_searching...")
                if self.exist_fixed_arc:
                    gff_out = []
                    for out_idx in range(self.num_layers):
                        if gff_arc[out_idx]:
                            gff_out.append(cell_outputs[out_idx])
                    gff_out.append(cell_outputs[-1])
                    num_filter = len(gff_out)*out_filters
                    gff_out = tf.concat(gff_out,3)
                    
                    print("Layer gff_out: {}".format(gff_out))
                    with tf.variable_scope("global_concat"):
                        w = create_weight("w_gc", [1, 1, num_filter, self.out_filters])
                        b = create_bias("b_gc", [self.out_filters])
                        x = tf.nn.conv2d(
                            gff_out, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                        w = create_weight("w_gfe", [3, 3, self.out_filters, self.sfe_filters])
                        b = create_bias("b_gfe", [self.sfe_filters])
                        x = tf.nn.conv2d(
                            x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                        print("Layer x_gfe: {}".format(x))
                else:
                    gff_arc = tf.concat([gff_arc, [tf.constant(1)]], 0)
                    gff_out=cell_outputs
                    indices = tf.where(tf.equal(gff_arc, 1))
                    indices = tf.to_int32(indices)
                    indices = tf.reshape(indices, [-1])
                    num_filter = tf.size(indices)*out_filters
                    gff_out = tf.gather(gff_out, indices, axis=0)
                    # gff_out.append(cell_outputs[-1])
                    inp = cell_outputs[-1]
                    if self.data_format == "NHWC":
                        N = tf.shape(inp)[0]
                        H = tf.shape(inp)[1]
                        W = tf.shape(inp)[2]
                        gff_out = tf.transpose(gff_out, [1, 2, 3, 0, 4])
                        gff_out = tf.reshape(gff_out, [N, H, W, num_filter])
                    with tf.variable_scope("global_concat"):
                        w = create_weight("w_gc", [self.num_layers + 1, 1 * 1 * out_filters * out_filters])
                        w = tf.gather(w, indices, axis=0)
                        w = tf.reshape(w, [1, 1, num_filter, out_filters])
                        b = create_bias("b_gc", [out_filters])
                        x = tf.nn.conv2d(
                            gff_out, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                        w = create_weight("w_gfe", [3, 3, self.out_filters, self.sfe_filters])
                        b = create_bias("b_gfe", [self.sfe_filters])
                        x = tf.nn.conv2d(
                            x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                        print("Layer x_gfe: {}".format(x))

            else:
                x_g_concat = tf.concat(cell_outputs, axis = 3)
                with tf.variable_scope("global_concat"):
                    w = create_weight("w_gc", [1, 1, self.out_filters*(self.num_layers+1), self.out_filters])
                    b = create_bias("b_gc", [self.out_filters])
                    x = tf.nn.conv2d(
                        x_g_concat, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                    w = create_weight("w_gfe", [3, 3, self.out_filters, self.sfe_filters])
                    b = create_bias("b_gfe", [self.sfe_filters])
                    x = tf.nn.conv2d(
                        x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                    # x = tf.Print(x, [1], message="x_gfe : ")
                    print("Layer x_gfe: {}".format(x))

            x = x + x_grl
            if self.upsample_size == 4:
                w = create_weight("w_rsu1", [3, 3, self.sfe_filters, 2 * 2 * 64])
                b = create_bias("b_rsu1", [2* 2 * 64])
                x = tf.nn.conv2d(
                    x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                x = tf.nn.depth_to_space(x, 2, data_format=self.data_format)
                w = create_weight("w_rsu2", [3, 3, 64, 2 * 2 * 64])
                b = create_bias("b_rsu2", [2* 2 * 64])
                x = tf.nn.conv2d(
                    x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                x_up_feature = tf.nn.depth_to_space(x, 2, data_format=self.data_format)
            else:
                w = create_weight("w_rsu", [3, 3, self.sfe_filters, self.upsample_size * self.upsample_size * 64])
                b = create_bias("b_rsu", [self.upsample_size * self.upsample_size * 64])
                x = tf.nn.conv2d(
                    x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                x_up_feature = tf.nn.depth_to_space(x, self.upsample_size, data_format=self.data_format)

            print("x_up_feature = {}".format(x_up_feature))
            with tf.variable_scope("result_conv"):
                inp_c = self._get_C(x_up_feature)
                w = create_weight("w", [3, 3, inp_c, 3])
                b = create_bias("b", [3])
                x = tf.nn.conv2d(x_up_feature, w, [1, 1, 1, 1], "SAME",
                                 data_format=self.data_format) + b
                # x = tf.Print(x, [1], message="final_x : ")
                print("Layer final_x: {}".format(x))


        return x

    def _model_CARN(self, images, is_training, reuse=False):
        """Compute the predictions given the images."""
        # self.now_arc = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        with tf.variable_scope(self.name, reuse=reuse):
            # the first two inputs
            with tf.variable_scope("stem_conv"):
                # w = create_weight("w", [3, 3, 3, self.out_filters * 3])
                w = create_weight("w_grl", [3, 3, 3, self.sfe_filters])
                b = create_bias("b_grl", [self.sfe_filters])
                x_grl = tf.nn.conv2d(
                    images, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                print("Layer x_grl: {}".format(x_grl))
                w = create_weight("w_sfe", [3, 3, self.sfe_filters, self.out_filters])
                b = create_weight("b_sfe", [self.out_filters])
                x_sfe = tf.nn.conv2d(
                    x_grl, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                # x_sfe = tf.Print(x_sfe, [1], message="x_sfe : ")
                print("Layer x_sfe: {}".format(x_sfe))

            if self.data_format == "NHWC":
                split_axis = 3
            elif self.data_format == "NCHW":
                split_axis = 1
            else:
                raise ValueError("Unknown data_format '{0}'".format(self.data_format))
            layers = [x_sfe]
            cell_outputs = []
            cell_outputs.append(x_sfe)
            # building layers in the micro space
            out_filters = self.out_filters
            for layer_id in range(self.num_layers):
                with tf.variable_scope("layer_{0}".format(layer_id)):
                    if self.exist_fixed_arc:
                        x = self._fixed_block(layers, self.fixed_arc, out_filters, 1, is_training)
                    else:
                        x = self._dnas_block(layers, self.fixed_arc, out_filters)
                    # x = tf.Print(x, [1], message="x : ")
                    print("Layer {0:>2d}: {1}".format(layer_id, x))
                    cell_outputs.append(x)
                    with tf.variable_scope("block_connection_{}".format(layer_id)):
                        next_inp = tf.concat([x_sfe,x], 3)
                        w = create_weight("w_bc", [1, 1, out_filters * 2, out_filters])
                        b = create_bias("b_bc", [out_filters])
                        layers = [tf.nn.conv2d(
                            next_inp, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b]

            # print("Layers in cell_outputs: {}".format(cell_outputs))
            if self.feature_fusion:
                gff_arc = self.fixed_arc[-self.num_layers:]
                print("feature_fusion_searching...")
                if self.exist_fixed_arc:
                    gff_out = []
                    for out_idx in range(self.num_layers):
                        if gff_arc[out_idx]:
                            gff_out.append(cell_outputs[out_idx])
                    gff_out.append(cell_outputs[-1])
                    num_filter = len(gff_out) * out_filters
                    gff_out = tf.concat(gff_out, 3)

                    print("Layer gff_out: {}".format(gff_out))
                    with tf.variable_scope("global_concat"):
                        w = create_weight("w_gc", [1, 1, num_filter, self.out_filters])
                        b = create_bias("b_gc", [self.out_filters])
                        x = tf.nn.conv2d(
                            gff_out, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                        w = create_weight("w_gfe", [3, 3, self.out_filters, self.sfe_filters])
                        b = create_bias("b_gfe", [self.sfe_filters])
                        x = tf.nn.conv2d(
                            x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                        print("Layer x_gfe: {}".format(x))
                else:
                    gff_arc = tf.concat([gff_arc, [tf.constant(1)]], 0)
                    gff_out = cell_outputs
                    indices = tf.where(tf.equal(gff_arc, 1))
                    indices = tf.to_int32(indices)
                    indices = tf.reshape(indices, [-1])
                    num_filter = tf.size(indices) * out_filters
                    gff_out = tf.gather(gff_out, indices, axis=0)
                    # gff_out.append(cell_outputs[-1])
                    inp = cell_outputs[-1]
                    if self.data_format == "NHWC":
                        N = tf.shape(inp)[0]
                        H = tf.shape(inp)[1]
                        W = tf.shape(inp)[2]
                        gff_out = tf.transpose(gff_out, [1, 2, 3, 0, 4])
                        gff_out = tf.reshape(gff_out, [N, H, W, num_filter])
                    with tf.variable_scope("global_concat"):
                        w = create_weight("w_gc", [self.num_layers + 1, 1 * 1 * out_filters * out_filters])
                        w = tf.gather(w, indices, axis=0)
                        w = tf.reshape(w, [1, 1, num_filter, out_filters])
                        b = create_bias("b_gc", [out_filters])
                        x = tf.nn.conv2d(
                            gff_out, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                        w = create_weight("w_gfe", [3, 3, self.out_filters, self.sfe_filters])
                        b = create_bias("b_gfe", [self.sfe_filters])
                        x = tf.nn.conv2d(
                            x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                        print("Layer x_gfe: {}".format(x))

            else:
                x_g_concat = tf.concat(cell_outputs, axis=3)
                with tf.variable_scope("global_concat"):
                    w = create_weight("w_gc", [1, 1, self.out_filters * (self.num_layers + 1), self.out_filters])
                    b = create_bias("b_gc", [self.out_filters])
                    x = tf.nn.conv2d(
                        x_g_concat, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                    w = create_weight("w_gfe", [3, 3, self.out_filters, self.sfe_filters])
                    b = create_bias("b_gfe", [self.sfe_filters])
                    x = tf.nn.conv2d(
                        x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                    # x = tf.Print(x, [1], message="x_gfe : ")
                    print("Layer x_gfe: {}".format(x))

            x = x + x_grl
            if self.upsample_size == 4:
                w = create_weight("w_rsu1", [3, 3, self.sfe_filters, 2 * 2 * 64])
                b = create_bias("b_rsu1", [2 * 2 * 64])
                x = tf.nn.conv2d(
                    x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                x = tf.nn.depth_to_space(x, 2, data_format=self.data_format)
                w = create_weight("w_rsu2", [3, 3, 64, 2 * 2 * 64])
                b = create_bias("b_rsu2", [2 * 2 * 64])
                x = tf.nn.conv2d(
                    x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                x_up_feature = tf.nn.depth_to_space(x, 2, data_format=self.data_format)
            else:
                w = create_weight("w_rsu", [3, 3, self.sfe_filters, self.upsample_size * self.upsample_size * 64])
                b = create_bias("b_rsu", [self.upsample_size * self.upsample_size * 64])
                x = tf.nn.conv2d(
                    x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
                x_up_feature = tf.nn.depth_to_space(x, self.upsample_size, data_format=self.data_format)

            print("x_up_feature = {}".format(x_up_feature))
            with tf.variable_scope("result_conv"):
                inp_c = self._get_C(x_up_feature)
                w = create_weight("w", [3, 3, inp_c, 3])
                b = create_bias("b", [3])
                x = tf.nn.conv2d(x_up_feature, w, [1, 1, 1, 1], "SAME",
                                 data_format=self.data_format) + b
                # x = tf.Print(x, [1], message="final_x : ")
                print("Layer final_x: {}".format(x))

        return x

    def _model_srcnn(self, images, is_training, reuse=False):
        """Compute the predictions given the images."""

        with tf.variable_scope(self.name, reuse=reuse):
            # the first two inputs
            with tf.variable_scope("srcnn"):
                # w = create_weight("w", [3, 3, 3, self.out_filters * 3])
                w = create_weight("w1", [9, 9, 3, 64])
                x = tf.nn.conv2d(images, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
                x = tf.nn.relu(x)
                w = create_weight("w2", [5, 5, 64, 32])
                x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
                x = tf.nn.relu(x)
                # w = create_weight("w3", [5, 5, 32, 3])
                # x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
                x_up_feature = tf.nn.depth_to_space(x, self.upsample_size, data_format=self.data_format)
                print("x_up_feature = {}".format(x_up_feature))
                with tf.variable_scope("result_conv"):
                    inp_c = self._get_C(x_up_feature)
                    w = create_weight("w3", [3, 3, inp_c, 3])
                    x = tf.nn.conv2d(x_up_feature, w, [1, 1, 1, 1], "SAME",
                                     data_format=self.data_format)
        return x

    def _model_RDN(self, images, is_training, reuse=False):

        scale = 2
        D = self.num_layers
        C = self.num_cells
        G = self.out_filters
        G0 = self.out_filters
        ks = 3
        c_dim = 3
        with tf.variable_scope(self.name, reuse=reuse):
            w_D_1 = create_weight("w_D_1", [1, 1, G * D, G0])
            w_D_2 = create_weight("w_D_2", [ks, ks, G0, G0])
            b_D_1 = create_bias("b_D_1", [G0])
            b_D_2 = create_bias("b_D_2", [G0])

            w_S_1 = create_weight("w_S_1", [ks, ks, c_dim, G0])
            w_S_2 = create_weight("w_S_2", [ks, ks, G0, G])
            b_S_1 = create_bias("b_S_1", [G0])
            b_S_2 = create_bias("b_S_2", [G])

            # weightsD = {
            #     'w_D_1': create_weight("w_D_1", [1, 1, G * D, G0]),
            #     'w_D_2': create_weight("w_D_2", [ks, ks, G0, G0])}
            # biasesD = {
            #     'b_D_1': create_bias("b_D_1", [G0]),
            #     'b_D_2': create_bias("b_D_2", [G0])}

            # weightsS = {
            #     'w_S_1': create_weight("w_S_1", [ks, ks, c_dim, G0]),
            #     'w_S_2': create_weight("w_S_2", [ks, ks, G0, G])}
            # biasesS = {
            #     'b_S_1': create_bias("b_S_1", [G0]),
            #     'b_S_2': create_bias("b_S_2", [G])}

            weight_final = create_weight("w_f", [ks, ks, G0/(scale*scale), c_dim])
            bias_final = create_bias("b_f", [c_dim])

            F_1 = tf.nn.conv2d(images, w_S_1, strides=[1, 1, 1, 1], padding='SAME') + b_S_1
            F0 = tf.nn.conv2d(F_1, w_S_2, strides=[1, 1, 1, 1], padding='SAME') + b_S_2
    
            FD = self._RDBs(F0,D,C, ks, G)
    
            FGF1 = tf.nn.conv2d(FD, w_D_1, strides=[1, 1, 1, 1], padding='SAME') + b_D_1
            FGF2 = tf.nn.conv2d(FGF1, w_D_2, strides=[1, 1, 1, 1], padding='SAME') + b_D_2
    
            FDF = tf.add(FGF2, F_1)
    
            # FU = self._UPN(FDF,scale, c_dim, G0)
            FU = tf.nn.depth_to_space(FDF,scale)
    
            IHR = tf.nn.conv2d(FU, weight_final, strides=[1, 1, 1, 1], padding='SAME') + bias_final

        return IHR

    # NOTE: train with batch size


    def _UPN(self, input_layer, scale, c_dim, G0):

        w_U_1 = create_weight("w_U_1", [5, 5, G0, 64])
        w_U_2 = create_weight("w_U_2", [3, 3, 64, 32])
        w_U_3 = create_weight("w_U_3", [3, 3, 32, c_dim * scale * scale])
        b_U_1 = create_bias("b_U_1", [64])
        b_U_2 = create_bias("b_U_2", [32])
        b_U_3 = create_bias("b_U_3", [c_dim * scale * scale])

        # weightsU = {
        #     'w_U_1': create_weight("w_U_1", [5, 5, G0, 64]),
        #     'w_U_2': create_weight("w_U_2", [3, 3, 64, 32]),
        #     'w_U_3': create_weight("w_U_3", [3, 3, 32, c_dim * scale * scale])}
        # biasesU = {
        #     'b_U_1': create_bias("b_U_1", [64]),
        #     'b_U_2': create_bias("b_U_2", [32]),
        #     'b_U_3': create_bias("b_U_3", [c_dim * scale * scale])}
        x = tf.nn.conv2d(input_layer, w_U_1, strides=[1, 1, 1, 1], padding='SAME') + b_U_1
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, w_U_2, strides=[1, 1, 1, 1], padding='SAME') + b_U_2
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, w_U_3, strides=[1, 1, 1, 1], padding='SAME') + b_U_3

        x = self._PS(x, scale, True)

        return x

    def _phase_shift(self, I, r):
        return tf.depth_to_space(I, r)

    def _PS(self, X, r, color=False):
        if color:
            Xc = tf.split(X, 3, 3)
            X = tf.concat([self._phase_shift(x, r) for x in Xc], 3)
        else:
            X = self._phase_shift(X, r)
        return X

    def _RDBs(self, input_layer, D, C, ks, G):
        # weightsR = {}
        # biasesR = {}
        # for i in range(1, D + 1):
        #     for j in range(1, C + 1):
        #         weightsR.update({'w_R_%d_%d' % (i, j): create_weight('w_R_%d_%d' % (i, j), [ks, ks, G * j, G])})
        #         biasesR.update({'b_R_%d_%d' % (i, j): create_bias('b_R_%d_%d' % (i, j), [G])})
        #     weightsR.update({'w_R_%d_%d' % (i, C + 1): create_weight('w_R_%d_%d' % (i, C + 1), [1, 1, G * (C + 1), G])})
        #     biasesR.update({'b_R_%d_%d' % (i, C + 1): create_bias('b_R_%d_%d' % (i, C + 1), [G])})
        rdb_concat = list()
        rdb_in = input_layer
        for i in range(1, D + 1):
            x = rdb_in
            for j in range(1, C + 1):
                weightsR = create_weight('w_R_%d_%d' % (i, j), [ks, ks, G * j, G])
                biasesR = create_bias('b_R_%d_%d' % (i, j), [G])
                tmp = tf.nn.conv2d(x, weightsR, strides=[1, 1, 1, 1], padding='SAME') + biasesR
                tmp = tf.nn.relu(tmp)
                x = tf.concat([x, tmp], axis=3)

            weightsR = create_weight('w_R_%d_%d' % (i, C + 1), [1, 1, G * (C + 1), G])
            biasesR = create_bias('b_R_%d_%d' % (i, C + 1), [G])
            x = tf.nn.conv2d(x, weightsR, strides=[1, 1, 1, 1], padding='SAME') + biasesR
            rdb_in = tf.add(x, rdb_in)
            print("Layer {0:>2d}: {1}".format(i, rdb_in))
            rdb_concat.append(rdb_in)


        return tf.concat(rdb_concat, axis=3)

    def _fixed_block(self, prev_layer, arc, out_filters, stride, is_training):
        """
        Args:
          prev_layer: cache of previous layer. for skip connections
          is_training: for batch_norm
        """
        assert len(prev_layer) == 1, "need exactly 1 inputs"
        layers = [prev_layer[0]]

        for cell_id in range(self.num_cells):
            start_id= int((cell_id / 2) * (2 * self.num_branches + (cell_id - 1) * self.num_branches))
            end_id = int(((cell_id+1) / 2) * (2 * self.num_branches + cell_id * self.num_branches))
            with tf.variable_scope("cell_{0}".format(cell_id)):
                x_connection = arc[start_id:end_id]
                x = self._fixed_cell(layers, cell_id, x_connection, out_filters, self.num_branches, stride, is_training)
                layers.append(x)
        # print("layers in fixed block: {}".format(layers))
        if self.feature_fusion:
            lff_arc = arc[-(self.num_cells+self.num_layers):-self.num_layers]
            lff_out = []
            for out_idx in range(self.num_cells):
                if lff_arc[out_idx]:
                    lff_out.append(layers[out_idx])
            lff_out.append(layers[-1])
            num_filter = len(lff_out) * out_filters
            lff_out = tf.concat(lff_out, 3)
            # print("Layer gff_out: {}".format(lff_out))
            with tf.variable_scope("local_concat"):
                w = create_weight("w_lc", [1, 1, num_filter, self.out_filters])
                b = create_bias("b_lc", [self.out_filters])
                out = tf.nn.conv2d(
                    lff_out, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
        else:
            x_l_concat = tf.concat(layers, axis=3)
            # print("Layer x_l_concat: {}".format(x_l_concat))
            with tf.variable_scope("local_concat"):
                w = create_weight("w_lc", [1, 1, self.out_filters*(self.num_cells+1), self.out_filters])
                b = create_bias("b_lc", [self.out_filters])
                out = tf.nn.conv2d(x_l_concat, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
        out = out + prev_layer[0]

        return out

    def _fixed_cell(self, prev_layers, cell_id, connection, out_filters, op_num, stride, is_training):
        """Performs an enas operation specified by op_id."""
        # op_connections = np.reshape(connection, [op_num, cell_id + 1])
        op_connections=[]
        conn_slice = np.reshape(connection,[cell_id+1,op_num])
        for i in range(op_num):
            op_connections.append(conn_slice[:,i])
                    
        op_exist = False
        cell_output = []
        if np.sum(op_connections[0]) > 0:
            seq = op_connections[0]
            inp = []
            for i in range(len(op_connections[0])):
                if seq[i]:
                    inp.append(prev_layers[i])
            inp = tf.concat(inp,3)

            out_conv = self._fixed_conv(inp, 3, out_filters, 1, is_training)
            op_exist = True
            cell_output.append(out_conv)
        else:
            out_conv = []

        if np.sum(op_connections[1]) > 0:
            seq = op_connections[1]
            inp = []
            for i in range(len(op_connections[1])):
                if seq[i]:
                    inp.append(prev_layers[i])
            inp = tf.concat(inp, 3)
            out_sep_conv = self._fixed_sep_conv(inp, 3, out_filters, 1, is_training)
            op_exist = True
            cell_output.append(out_sep_conv)
        else:
            out_sep_conv = []

        if np.sum(op_connections[2]) > 0:
            seq = op_connections[2]
            inp = []
            for i in range(len(op_connections[2])):
                if seq[i]:
                    inp.append(prev_layers[i])
            inp = tf.concat(inp, 3)
            out_dilated_conv = self._fixed_dilated_conv(inp, 3, 3, out_filters, is_training)
            op_exist = True
            cell_output.append(out_dilated_conv)
        else:
            out_dilated_conv = []
        out = tf.stack(cell_output,0)

        if op_exist:
            out = tf.reduce_mean(out, 0)
            out = tf.nn.relu(out)
        else:
            out = prev_layers[-1]
            out = tf.nn.relu(out)
        if self.channel_attn:
            with tf.variable_scope("channel_attn"):
                strides = self._get_strides(stride)
                c = out.get_shape()[-1]
                H_gp = tf.reshape(tf.reduce_mean(out, axis=[1, 2]), (-1, 1, 1, c))
                wd_weights = create_weight("wd_w", [1, 1, c, int(c)/4])
                wd_biases = create_bias("wd_b", [int(c)/4])
                W_d = tf.nn.conv2d(H_gp, wd_weights, strides, "SAME") + wd_biases
                W_d = tf.nn.relu(W_d)
                wu_weights = create_weight("wu_w", [1, 1, int(c)/4, c])
                wu_biases = create_bias("wu_b", [c])
                W_u = tf.nn.conv2d(W_d, wu_weights, strides, "SAME") + wu_biases
                f = tf.nn.sigmoid(W_u)
                out = tf.multiply(f,out)

                # H_gp = tf.reduce_mean(out, axis=[1,2])
                # wd_weights = create_weight("wd_weight", [out_filters, out_filters/4])
                # wd_biases = create_bias("wd_bias", [out_filters/4])
                # W_d = tf.matmul(H_gp, wd_weights) + wd_biases
                # W_d = tf.nn.relu(W_d)
                # wu_weights = create_weight("wu_weight", [out_filters/4, out_filters])
                # wu_biases = create_bias("wu_bias", [out_filters])
                # W_u = tf.matmul(W_d, wu_weights) + wu_biases
                # f = tf.math.sigmoid(W_u)
                # f = tf.reshape(f,[self.batch_size,1,1,out_filters])
                # out = f*out

        return out

    def _fixed_sep_conv(self, x, f_size, out_filters, stride, is_training):
        """Apply fixed convolution.

        Args:
          stacked_convs: number of separable convs to apply.
        """
        inp_c = self._get_C(x)
        strides = self._get_strides(stride)

        with tf.variable_scope("sep_conv"):
            w_depthwise = create_weight("w_depth", [f_size, f_size, inp_c, 1])
            w_pointwise = create_weight("w_point", [1, 1, inp_c, out_filters])
            b = create_bias("b", [out_filters])
            x = tf.nn.relu(x)
            x = tf.nn.separable_conv2d(
                x,
                depthwise_filter=w_depthwise,
                pointwise_filter=w_pointwise,
                strides=strides, padding="SAME", data_format=self.data_format) + b
            # x = batch_norm(x, is_training, data_format=self.data_format)

        return x

    def _fixed_dilated_conv(self, x, rate, f_size, out_filters, is_training):
        """Apply fixed convolution.

        Args:

        """

        inp_c = self._get_C(x)
        with tf.variable_scope("dilated_conv"):
            x = tf.nn.relu(x)
            w = create_weight("w", [f_size, f_size, inp_c, out_filters])
            b = create_bias("b", [out_filters])
            x = tf.nn.atrous_conv2d(x, filters=w, rate=rate, padding="SAME") + b
            # x = batch_norm(x, is_training, data_format=self.data_format)

        return x

    def _fixed_conv(self, x, f_size, out_filters, stride, is_training):
        """Apply fixed convolution.

        Args:

        """

        inp_c = self._get_C(x)
        strides = self._get_strides(stride)

        with tf.variable_scope("conv"):
            x = tf.nn.relu(x)
            w = create_weight("w", [f_size, f_size, inp_c, out_filters])
            b = create_bias("b", [out_filters])
            x = tf.nn.conv2d(x, w, strides, "SAME") + b
            # x = batch_norm(x, is_training, data_format=self.data_format)

        return x

    # def _dnas_cell(self, prev_layers, cell_id, connection, out_filters, op_num):
    #     """Performs an enas operation specified by op_id."""
    #     if tf.equal(tf.reduce_sum(connection),0):
    #         out = prev_layers[-1]
    #     else:
    #         op_connections = tf.reshape(connection,[op_num,cell_id+1])
    #         out = []
    #         x = self._dnas_conv(prev_layers, cell_id, op_connections[0], 3, out_filters)
    #         if x is not None:
    #             out.append(x)
    #         x = self._dnas_sep_conv(prev_layers, cell_id, op_connections[1], 3, out_filters)
    #         if x is not None:
    #             out.append(x)
    #         x = self._dnas_dilated_conv(prev_layers, cell_id, op_connections[2], 3, 3, out_filters)
    #         if x is not None:
    #             out.append(x)
    #         out = tf.reduce_mean(out,0)
    #         out = tf.nn.relu(out)
    #
    #     return out

    def _dnas_block(self, prev_layer, arc, out_filters):
        """
        Args:
          layer_id: current layer
          prev_layers: cache of previous layers. for skip connections
          start_idx: where to start looking at. technically, we can infer this
            from layer_id, but why bother...
        """

        assert len(prev_layer) == 1, "need exactly 1 inputs"
        layers = [prev_layer[0]]

        for cell_id in range(self.num_cells):
            start_id= int((cell_id / 2) * (2 * self.num_branches + (cell_id - 1) * self.num_branches))
            end_id = int(((cell_id+1) / 2) * (2 * self.num_branches + cell_id * self.num_branches))
            with tf.variable_scope("cell_{0}".format(cell_id)):
                x_connection = arc[start_id:end_id]
                x = self._dnas_cell(layers, cell_id, x_connection, out_filters, self.num_branches)
                layers.append(x)
        # print("layers in dnas block: {}".format(layers))

        if self.feature_fusion:
            lff_arc = arc[-(self.num_cells+self.num_layers):-self.num_layers]
            lff_arc = tf.concat([lff_arc, [tf.constant(1)]], 0)
            lff_out = layers
            indices = tf.where(tf.equal(lff_arc, 1))
            indices = tf.to_int32(indices)
            indices = tf.reshape(indices, [-1])
            num_filter = tf.size(indices) * out_filters
            lff_out = tf.gather(lff_out, indices, axis=0)
            print("lff_out : {}".format(lff_out))
            # lff_out.append(layers[-1])
            inp = layers[-1]
            if self.data_format == "NHWC":
                N = tf.shape(inp)[0]
                H = tf.shape(inp)[1]
                W = tf.shape(inp)[2]
                lff_out = tf.transpose(lff_out, [1, 2, 3, 0, 4])
                lff_out = tf.reshape(lff_out, [N, H, W, num_filter])
            with tf.variable_scope("local_concat"):
                w = create_weight("w_lc", [self.num_cells+1, 1*1*out_filters*out_filters])
                w = tf.gather(w, indices, axis=0)
                w = tf.reshape(w, [1,1, num_filter, out_filters])
                b = create_bias("b_lc",[out_filters])
                out = tf.nn.conv2d(
                    lff_out, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
        else:
            x_l_concat = tf.concat(layers, axis=3)
            # print("Layer x_l_concat: {}".format(x_l_concat))
            with tf.variable_scope("local_concat"):
                w = create_weight("w_lc", [1, 1, self.out_filters * (self.num_cells + 1), self.out_filters])
                b = create_bias("b_lc",[self.out_filters])
                out = tf.nn.conv2d(x_l_concat, w, [1, 1, 1, 1], "SAME", data_format=self.data_format) + b
        out = out + prev_layer[0]

        return out

    def _dnas_cell(self, prev_layers, cell_id, connection, out_filters, op_num):
        """Performs an enas operation specified by op_id."""
        # op_connections = tf.reshape(connection, [op_num, cell_id + 1])
        op_connections=[]
        conn_slice = tf.reshape(connection,[cell_id+1,op_num])
        for i in range(op_num):
            op_connections.append(conn_slice[:,i])
            
        op_exist = []
        cell_output = []

        out_conv, conv_exist = tf.cond(tf.reduce_sum(op_connections[0]) > 0,
                           lambda: self._dnas_conv(prev_layers, cell_id, op_connections[0], 3, out_filters),
                           lambda: self._dnas_dummy(prev_layers))
        out_sep_conv, sep_conv_exist = tf.cond(tf.reduce_sum(op_connections[1]) > 0,
                           lambda: self._dnas_sep_conv(prev_layers, cell_id, op_connections[1], 3, out_filters),
                           lambda: self._dnas_dummy(prev_layers))
        out_dilated_conv, dilated_conv_exist = tf.cond(tf.reduce_sum(op_connections[2]) > 0,
                           lambda: self._dnas_dilated_conv(prev_layers, cell_id, op_connections[2], 3, 3, out_filters),
                           lambda: self._dnas_dummy(prev_layers))
        print("out_conv : {}".format(out_conv))
        cell_output.append(out_conv)
        cell_output.append(out_sep_conv)
        cell_output.append(out_dilated_conv)
        op_exist.append(conv_exist)
        op_exist.append(sep_conv_exist)
        op_exist.append(dilated_conv_exist)
        op_exist=tf.reshape(op_exist, [-1])
        indices = tf.where(tf.equal(op_exist,True))
        indices = tf.to_int32(indices)
        indices = tf.reshape(indices, [-1])
        out = tf.gather(cell_output, indices, axis=0)
        out = tf.cond(tf.reduce_sum(connection)>0, lambda:self._get_dnas_cell_out(True, prev_layers, out), lambda:self._get_dnas_cell_out(False, prev_layers, out))

        if self.channel_attn:
            with tf.variable_scope("channel_attn"):
                strides = self._get_strides(1)
                c = out.get_shape()[-1]
                H_gp = tf.reshape(tf.reduce_mean(out, axis=[1, 2]), (-1, 1, 1, c))
                wd_weights = create_weight("wd_w", [1, 1, c, int(c)/4])
                wd_biases = create_bias("wd_b", [int(c)/4])
                W_d = tf.nn.conv2d(H_gp, wd_weights, strides, "SAME") + wd_biases
                W_d = tf.nn.relu(W_d)
                wu_weights = create_weight("wu_w", [1, 1, int(c)/4, c])
                wu_biases = create_bias("wu_b", [c])
                W_u = tf.nn.conv2d(W_d, wu_weights, strides, "SAME") + wu_biases
                f = tf.nn.sigmoid(W_u)
                out = tf.multiply(f,out)

        return out

    def _dnas_dummy(self, prev_layers):
        dummy_node=prev_layers[0]
        return dummy_node, False

    def _get_dnas_cell_out(self, exist, prev_layers, out):
        if exist:
            out = tf.reduce_mean(out, 0)
            out = tf.nn.relu(out)
        else:
            out = prev_layers[-1]
            out = tf.nn.relu(out)
        return out

    def _dnas_conv(self, prev_layers, cell_id, connections, filter_size, out_filters):
        """Performs an enas convolution specified by the relevant parameters."""

        with tf.variable_scope("conv_{0}".format(filter_size)):
            num_possible_inputs = cell_id + 1
            connections = tf.reshape(connections, [-1])
            indices = tf.where(tf.equal(connections, 1))
            indices = tf.to_int32(indices)
            indices = tf.reshape(indices, [-1])
            num_outs = tf.size(indices)

            x = tf.gather(prev_layers, indices, axis=0)
            # create params and pick the correct path
            inp = prev_layers[0]
            if self.data_format == "NHWC":
                N = tf.shape(inp)[0]
                H = tf.shape(inp)[1]
                W = tf.shape(inp)[2]
                C = tf.shape(inp)[3]
                x = tf.transpose(x, [1, 2, 3, 0, 4])
                x = tf.reshape(x, [N, H, W, num_outs * out_filters])
            w = create_weight("w", [num_possible_inputs, filter_size*filter_size*out_filters*out_filters])
            w = tf.gather(w, indices, axis=0)
            w = tf.reshape(w, [filter_size, filter_size, num_outs * out_filters, out_filters])
            b = create_bias("b",[out_filters])

            x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME")
            # x, _, _ = tf.nn.fused_batch_norm(
            #     x, scale, offset, epsilon=1e-5, data_format=self.data_format,
            #     is_training=True)
        return x, True

    def _dnas_sep_conv(self, prev_layers, cell_id, connections, filter_size, out_filters):
        """Performs an enas convolution specified by the relevant parameters."""

        with tf.variable_scope("sep_conv_{0}".format(filter_size)):
            num_possible_inputs = cell_id + 1
            connections = tf.reshape(connections, [-1])
            indices = tf.where(tf.equal(connections, 1))
            indices = tf.to_int32(indices)
            indices = tf.reshape(indices, [-1])
            num_outs = tf.size(indices)

            x = tf.gather(prev_layers, indices, axis=0)
            # create params and pick the correct path
            inp = prev_layers[0]
            if self.data_format == "NHWC":
                N = tf.shape(inp)[0]
                H = tf.shape(inp)[1]
                W = tf.shape(inp)[2]
                C = tf.shape(inp)[3]
                x = tf.transpose(x, [1, 2, 3, 0, 4])
                x = tf.reshape(x, [N, H, W, num_outs * out_filters])

            # create params and pick the correct path
            w_depthwise = create_weight(
                "w_depth", [num_possible_inputs, filter_size * filter_size * out_filters])
            w_depthwise = tf.gather(w_depthwise, indices, axis=0)
            w_depthwise = tf.reshape(
                w_depthwise, [filter_size, filter_size, out_filters*num_outs, 1])

            w_pointwise = create_weight(
                "w_point", [num_possible_inputs, out_filters * out_filters])
            w_pointwise = tf.gather(w_pointwise, indices, axis=0)
            w_pointwise = tf.reshape(w_pointwise, [1, 1, out_filters*num_outs, out_filters])
            b = create_bias("b", [out_filters])

            x = tf.nn.separable_conv2d(
                x,
                depthwise_filter=w_depthwise,
                pointwise_filter=w_pointwise,
                strides=[1, 1, 1, 1], padding="SAME",
                data_format=self.data_format) + b
        return x, True

    def _dnas_dilated_conv(self, prev_layers, cell_id, connections, rate, filter_size, out_filters):
        """Performs an enas convolution specified by the relevant parameters."""

        with tf.variable_scope("dilated_conv_{}_{}".format(filter_size,rate)):
            num_possible_inputs = cell_id + 1
            connections = tf.reshape(connections, [-1])
            indices = tf.where(tf.equal(connections, 1))
            indices = tf.to_int32(indices)
            indices = tf.reshape(indices, [-1])
            num_outs = tf.size(indices)
            # create params and pick the correct path
            inp = prev_layers[0]
            outs=[]
            # tensor_for_input_size = prev_layers
            if self.data_format == "NHWC":
                N = tf.shape(inp)[0]
                H = tf.shape(inp)[1]
                W = tf.shape(inp)[2]
                C = tf.shape(inp)[3]
                # outs = tf.zeros_like(tensor_for_input_size, dtype=tf.float32)
                # print("prev_layer[0] size : {}".format(inp))
                # print("outs in dnas_dilated_conv : {}".format(outs))
                for i in range(num_possible_inputs):
                    out = tf.cond(tf.equal(connections[i], 1), lambda: prev_layers[i], lambda:tf.zeros_like(prev_layers[i],dtype=tf.float32))
                    outs.append(out)

                # print("layers in dnas_dilated_conv outs: {}".format(outs))
                x = tf.stack(outs, axis=0)
                # print("layers in dnas_dilated_conv x.stack: {}".format(x))
                x = tf.transpose(x, [1, 2, 3, 0, 4])
                # print("layers in dnas_dilated_conv x.transpose: {}".format(x))
                x = tf.reshape(x, [N, H, W, num_possible_inputs * out_filters])
                # print("layers in dnas_dilated_conv x.reshape: {}".format(x))
            w = create_weight("w", [filter_size,filter_size,num_possible_inputs*out_filters,out_filters])
            b = create_bias("b", [out_filters])
            x = tf.nn.atrous_conv2d(x, filters=w, rate=rate, padding="SAME") + b

            # x, _, _ = tf.nn.fused_batch_norm(
            #     x, scale, offset, epsilon=1e-5, data_format=self.data_format,
            #     is_training=True)
        return x, True

    # override
    def _build_train(self):
        print("-" * 80)
        print("Build train graph")
        if self.use_model == "SRCNN":
            self.train_preds = self._model_srcnn(self.x_train, True)
        elif self.use_model == "RDN":
            self.train_preds = self._model_RDN(self.x_train, True)
        elif self.use_model == "CARN":
            self.train_preds = self._model_CARN(self.x_train, True)
        else:
            self.train_preds = self._model(self.x_train, True)
        self.loss = tf.losses.absolute_difference(labels=self.y_train, predictions=self.train_preds)
        # self.loss = tf.Print(self.loss, [self.loss], message="self.loss changed! : ")
        # self.loss = tf.losses.mean_squared_error(labels=self.y_train, predictions=self.train_preds)

        # self.train_PSNR = tf.image.psnr(self.y_valid, preds, 1)
        # self.train_PSNR = tf.reduce_sum(self.valid_PSNR)
        # self.aux_loss = tf.losses.mean_squared_error(labels=self.y_train, predictions=self.aux_preds)
        if self.use_aux_heads:
            # train_loss = self.loss + 0.4 * self.aux_loss
            train_loss = self.loss
        else:
            train_loss = self.loss

        tf_variables = [
            var for var in tf.trainable_variables() if (
                    var.name.startswith(self.name) and "aux_head" not in var.name)]
        self.num_vars = count_model_params(tf_variables)
        print("Model has {0} params".format(self.num_vars))

        self.train_op, self.lr, self.grad_norm, self.optimizer, self.grads = get_train_ops(
            train_loss,
            tf_variables,
            self.global_step,
            clip_mode=self.clip_mode,
            grad_bound=self.grad_bound,
            l2_reg=self.l2_reg,
            lr_init=self.lr_init,
            lr_dec_start=self.lr_dec_start,
            lr_warmup_steps = self.lr_warmup_steps,
            lr_warmup_val = self.lr_warmup_val,            
            lr_dec_rate=self.lr_dec_rate,
            lr_dec_every=self.lr_dec_every,
            lr_dec_min=self.lr_dec_min,
            lr_cosine=self.lr_cosine,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            lr_T_0=self.lr_T_0,
            lr_T_mul=self.lr_T_mul,
            num_train_batches=self.num_train_batches,
            optim_algo=self.optim_algo
            # sync_replicas=self.sync_replicas,
            # num_aggregate=self.num_aggregate,
            # num_replicas=self.num_replicas
        )

    # override
    def _build_valid(self):
        if self.x_valid is not None:
            print("-" * 80)
            print("Build valid graph")
            if self.use_model == "SRCNN":
                self.valid_preds = self._model_srcnn(self.x_valid, False, reuse=True)
            elif self.use_model == "RDN":
                self.valid_preds = self._model_RDN(self.x_valid, False, reuse=True)
            elif self.use_model == "CARN":
                self.valid_preds = self._model_CARN(self.x_valid, False, reuse=True)
            else:
                self.valid_preds = self._model(self.x_valid, False, reuse=True)
            # self.loss = tf.losses.mean_squared_error(labels=self.y_valid, predictions=self.valid_preds)
            # self.valid_PSNR = tf.image.psnr(self.y_valid, preds, 1)
            # self.valid_PSNR = tf.reduce_sum(self.valid_PSNR)

    # override
    def _build_test(self):
        print("-" * 80)
        print("Build test graph")
        if self.use_model == "SRCNN":
            self.test_preds = self._model_srcnn(self.x_test, False, reuse=True)
        elif self.use_model == "RDN":
            self.test_preds = self._model_RDN(self.x_test, False, reuse=True)
        elif self.use_model == "CARN":
            self.test_preds = self._model_CARN(self.x_test, False, reuse=True)
        else:
            self.test_preds = self._model(self.x_test, False, reuse=True)
        # self.loss = tf.losses.mean_squared_error(labels=self.y_test, predictions=self.test_preds)
        # self.test_PSNR = tf.image.psnr(self.y_test, self.test_preds, 1)
        # self.test_PSNR = tf.reduce_sum(self.test_PSNR)

    # override
    def build_valid_rl(self, shuffle=False):
        if self.x_valid_rl is not None:
            print("-" * 80)
            print("Build valid graph for rl")
            if self.use_model == "SRCNN":
                self.valid_preds_rl = self._model_srcnn(self.x_valid_rl, False, reuse=True)
            elif self.use_model == "RDN":
                self.valid_preds_rl = self._model_RDN(self.x_valid_rl, False, reuse=True)
            elif self.use_model == "CARN":
                self.valid_preds_rl = self._model_CARN(self.x_valid_rl, False, reuse=True)

            else:
                self.valid_preds_rl = self._model(self.x_valid_rl, False, reuse=True)
            self.valid_rl_PSNR = tf.Variable(0.,dtype=tf.float32)
            # self.loss = tf.losses.mean_squared_error(labels=self.y_valid_rl, predictions=self.valid_preds_rl)
            # self.valid_rl_PSNR = tf.image.psnr(self.y_valid_rl, self.valid_preds_rl, 1)
            # self.valid_rl_PSNR = tf.reduce_sum(self.valid_rl_PSNR)

    def connect_controller(self, controller_model):
        if self.exist_fixed_arc:
            self.fixed_arc = np.array([int(x) for x in self.fixed_arc.split(" ") if x])
        else:
            # self.fixed_arc = controller_model.sample_arc
            self.now_arc = tf.placeholder(dtype=tf.int32)
            self.fixed_arc = self.now_arc
            # self.fixed_arc = tf.Print(self.fixed_arc, [self.fixed_arc], message="now arc is:",summarize=-1)

        self._build_train()
        self._build_valid()
        self._build_test()
