import os
import shutil
import sys
import time
import glob

import numpy as np
import tensorflow as tf
import cv2

from src import utils
from src.utils import Logger
from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
from src.utils import print_user_flags
from src.utils import calculate_cb_penalty

from src.utils_img import imresize
from src.utils_img import bgr2y
from src.utils_img import calc_psnr
from src.utils_img import modcrop
from src.utils_img import calculate_psnr
from src.utils_img import calculate_ssim

from src.DIV2K.data_utils import threaded_input_word_pipeline
from src.DIV2K.data_utils import make_eval_batch
from src.DIV2K.data_utils import get_random_batch

from src.DIV2K.controller import ControllerNetwork
from src.DIV2K.controller_inference import ControllerInferenceNetwork
from src.DIV2K.child_network import ChildNetwork

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

flags = tf.app.flags
FLAGS = flags.FLAGS
# parameters for result, in/out data
DEFINE_boolean("reset_output_dir", True, "Delete output_dir if exists.")
DEFINE_string("data_path", "../tfrecord/", "path of train,valid tfrecord folder")
DEFINE_string("img_path", "../data/", "path of test image folder")
DEFINE_string("data_format", "NHWC", "image data format. 'NHWC' or 'NCHW'")
DEFINE_string("output_dir", "./outputs/fixed_small_cb_x4", "path of result")
DEFINE_string("checkpoint", "model.ckpt-931000", "path of checkpoint file")
DEFINE_string("checkpoint_dir", "./outputs/x2", "path of checkpoint file")

DEFINE_boolean("test_mode", False, "use when test")
DEFINE_boolean("inference_mode", False, "use when inference")
DEFINE_string("use_model", None, "which model to use for training")
DEFINE_boolean("rl_search", False, "use global/local feature fusion searching")
DEFINE_boolean("cb_reward", True, "use complexity based reward")
DEFINE_float("cb_rate", 2, "rate of complexity based reward")

# parameters for batch and training
DEFINE_integer("batch_size", 4, "batch size in training process")
DEFINE_integer("num_epochs", 600, "training epoch for child_network")
DEFINE_integer("it_per_epoch", 1000, "iteration of 1 epoch for child_network")
DEFINE_integer("eval_batch_size", 20, "batch size of evaluation process")
DEFINE_integer("test_batch_size", 1, "batch size of test process")
DEFINE_integer("loss_cut", 2, "cut training process when loss > avgLoss*loss_cut")
DEFINE_boolean("image_random", False, "use when test")
DEFINE_boolean("channel_attn", True, "use channel_attn method or not")

# parameters for child_network design
DEFINE_integer("child_upsample_size", 2, "rate of lr image size")
DEFINE_integer("child_num_layers", 4, "number of Cells")
DEFINE_integer("child_num_cells", 4, "number of layers in cells")
DEFINE_integer("child_out_filters", 64, "number of out filter channels of each cells")
DEFINE_integer("child_num_sfe", 64, "number of out filter channels of shallow feature extraction layer")
DEFINE_integer("child_num_branches", 3, "number of operations in search space")
DEFINE_string("child_fixed_arc", "1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 1 0 0 1 0 0 1 1 1 0 0 0 0 1", "")
DEFINE_boolean("child_use_aux_heads", False, "Should we use an aux head")
DEFINE_boolean("child_sync_replicas", False, "To sync or not to sync.")
DEFINE_integer("child_num_aggregate", None, "")
DEFINE_integer("child_num_replicas", 1, "")

# DEFINE_integer("child_block_size", 3, "")
# DEFINE_integer("child_out_filters_scale", 1, "")
# DEFINE_integer("child_filter_size", 5, ""

# parameters for child_network learning rate, gradient, loss
DEFINE_integer("child_lr_dec_every", 50, "learning rate decay step size of child network")
DEFINE_integer("child_cutout_size", None, "CutOut size")
DEFINE_float("child_grad_bound", None, "Gradient clipping")
DEFINE_float("child_lr", 1e-4, "")
DEFINE_float("child_lr_dec_rate", 0.5, "")
DEFINE_float("child_lr_dec_min", 1e-12, "")
DEFINE_float("child_l2_reg", 0, "")
DEFINE_float("child_lr_warmup_val", None, "warming up learning rate")
DEFINE_integer("child_lr_warmup_step", 0, "step to use warmup learning rate")
DEFINE_integer("child_lr_dec_start", 0, "step to start learning rate decrease")
# parameters for child_network cosine lr
DEFINE_boolean("child_lr_cosine", False, "Use cosine lr schedule")
DEFINE_float("child_lr_max", 0.05, "for lr schedule")
DEFINE_float("child_lr_min", 0.0005, "for lr schedule")
DEFINE_integer("child_lr_T_0", 10, "for lr schedule")
DEFINE_integer("child_lr_T_mul", 2, "for lr schedule")

# parameters for controller
DEFINE_float("controller_lr", 0.0003, "")
DEFINE_float("controller_lr_dec_rate", 1.0, "")
DEFINE_float("controller_l2_reg", 0.0, "")
DEFINE_float("controller_bl_dec", 0.99, "")
DEFINE_float("controller_tanh_constant", 1.10, "")
DEFINE_float("controller_op_tanh_reduce", 2.5, "")
DEFINE_float("controller_temperature", None, "")
DEFINE_float("controller_entropy_weight", 0.0001, "")
DEFINE_float("controller_skip_target", 0.8, "")
DEFINE_float("controller_skip_weight", 0.0, "")
DEFINE_integer("controller_num_aggregate", None, "")
DEFINE_integer("controller_num_replicas", 1, "")
DEFINE_integer("controller_train_steps", 100, "")
DEFINE_integer("controller_train_every", 1, "train the controller after this number of epochs")
DEFINE_integer("controller_train_start", 5, "start controller training epoch")
DEFINE_float("controller_best_rate", 0, "rate of training controller by best architecture")

DEFINE_boolean("controller_sync_replicas", False, "To sync or not to sync.")
DEFINE_boolean("controller_training", False, "")

DEFINE_float("controller_keep_prob", 0.5, "")
DEFINE_integer("controller_forwards_limit", 2, "")
DEFINE_boolean("controller_use_critic", False, "")

DEFINE_integer("log_every", 200, "How many steps to log")
DEFINE_integer("controller_log_every", 20, "How many steps to log when training controller")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")

# parameters for ???
# DEFINE_float("child_keep_prob", 0.90, "")
# DEFINE_float("child_drop_path_keep_prob", 0.60, "minimum drop_path_keep_prob")
DEFINE_string("child_skip_pattern", None, "Must be ['dense', None]")


def evaluate():
    valid_file_path = os.path.join(FLAGS.data_path, 'DIV2K_valid_x2')
    test_set =['Set5', 'Set14', 'BSDS100', 'Urban100']
    test_file_paths = []
    for i in range(4):
        test_file_paths.append(os.path.join(FLAGS.img_path, test_set[i]))
    # test_file_path = os.path.join(FLAGS.img_path, 'Set5')
    bgr_mean = np.array([103.154 / 255, 111.561 / 255, 114.356 / 255]).astype(np.float32)
    images = {}
    labels = {}
    meta_data = {}
    g = tf.Graph()
    with g.as_default():
        images["valid"], labels["valid"], meta_data["valid"] = threaded_input_word_pipeline(valid_file_path,
                                                                                            file_patterns=[
                                                                                                '*.tfrecord'],
                                                                                            num_threads=4,
                                                                                            batch_size=FLAGS.eval_batch_size,
                                                                                            img_size=48,
                                                                                            label_size=96,
                                                                                            num_epochs=None,
                                                                                            is_train=False)
        for i in range(4):
            images[test_set[i]], labels[test_set[i]], meta_data[test_set[i]] = make_eval_batch(test_file_paths[i], 'x2_small', FLAGS.child_upsample_size)

        images["test"], labels["test"], meta_data["test"] = make_eval_batch(test_file_paths[0], 'x2_small', FLAGS.child_upsample_size)
        images["valid_rl"], labels["valid_rl"], meta_data["valid_rl"] = None, None, None

        print("data_num of test data = {}".format(meta_data[test_set[i]]["total_data_num"]))

        print("build controller, child_network...")
        controllerClass = ControllerNetwork
        childClass = ChildNetwork
        child_model = childClass(
            images,
            labels,
            meta_data,
            output_dir=FLAGS.output_dir,
            use_aux_heads=FLAGS.child_use_aux_heads,
            use_model=FLAGS.use_model,
            feature_fusion=FLAGS.rl_search,
            channel_attn=FLAGS.channel_attn,            
            cb_reward=FLAGS.cb_reward,
            cutout_size=FLAGS.child_cutout_size,
            num_layers=FLAGS.child_num_layers,
            num_cells=FLAGS.child_num_cells,
            num_branches=FLAGS.child_num_branches,
            fixed_arc=FLAGS.child_fixed_arc,
            out_filters=FLAGS.child_out_filters,
            upsample_size=FLAGS.child_upsample_size,
            sfe_filters=FLAGS.child_num_sfe,
            num_epochs=FLAGS.num_epochs,
            it_per_epoch=FLAGS.it_per_epoch,
            l2_reg=FLAGS.child_l2_reg,
            data_format=FLAGS.data_format,
            batch_size=FLAGS.batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            test_batch_size=FLAGS.test_batch_size,
            clip_mode=None,
            grad_bound=FLAGS.child_grad_bound,
            lr_init=FLAGS.child_lr,
            lr_dec_every=FLAGS.child_lr_dec_every,
            lr_dec_rate=FLAGS.child_lr_dec_rate,
            lr_dec_min=FLAGS.child_lr_dec_min,
            lr_warmup_val=FLAGS.child_lr_warmup_val,
            lr_warmup_step=FLAGS.child_lr_warmup_step,
            lr_dec_start=FLAGS.child_lr_dec_start,
            lr_cosine=FLAGS.child_lr_cosine,
            lr_max=FLAGS.child_lr_max,
            lr_min=FLAGS.child_lr_min,
            lr_T_0=FLAGS.child_lr_T_0,
            lr_T_mul=FLAGS.child_lr_T_mul,
            optim_algo="adam",
            sync_replicas=FLAGS.child_sync_replicas,
            num_aggregate=FLAGS.child_num_aggregate,
            num_replicas=FLAGS.child_num_replicas,
        )

        if FLAGS.child_fixed_arc is None:
            print("fixed arc is None. training controllers.")
            controller_model = controllerClass(
                feature_fusion=FLAGS.rl_search,
                use_cb_reward=FLAGS.cb_reward,
                cb_rate=FLAGS.cb_rate,
                skip_target=FLAGS.controller_skip_target,
                skip_weight=FLAGS.controller_skip_weight,
                num_cells=FLAGS.child_num_cells,
                num_layers=FLAGS.child_num_layers,
                num_branches=FLAGS.child_num_branches,
                out_filters=FLAGS.child_out_filters,
                lstm_size=64,
                lstm_num_layers=2,
                lstm_keep_prob=1.0,
                tanh_constant=FLAGS.controller_tanh_constant,
                op_tanh_reduce=FLAGS.controller_op_tanh_reduce,
                temperature=FLAGS.controller_temperature,
                lr_init=FLAGS.controller_lr,
                lr_dec_start=0,
                lr_dec_every=1000000,  # never decrease learning rate
                l2_reg=FLAGS.controller_l2_reg,
                entropy_weight=FLAGS.controller_entropy_weight,
                bl_dec=FLAGS.controller_bl_dec,
                use_critic=FLAGS.controller_use_critic,
                optim_algo="adam",
                sync_replicas=FLAGS.controller_sync_replicas,
                num_aggregate=FLAGS.controller_num_aggregate,
                num_replicas=FLAGS.controller_num_replicas)

            child_model.connect_controller(controller_model)
            controller_model.build_trainer(child_model)

            controller_ops = {
                "train_step": controller_model.train_step,
                "loss": controller_model.loss,
                "train_op": controller_model.train_op,
                "lr": controller_model.lr,
                "grad_norm": controller_model.grad_norm,
                "valid_PSNR": controller_model.valid_PSNR,
                "optimizer": controller_model.optimizer,
                "baseline": controller_model.baseline,
                "entropy": controller_model.sample_entropy,
                "sample_arc": controller_model.sample_arc,
                "skip_rate": controller_model.skip_rate,
                "batch_size": child_model.batch_size,
                "reward": controller_model.reward,
                "log_prob": controller_model.sample_log_prob,
            }

        else:
            assert not FLAGS.controller_training, (
                "--child_fixed_arc is given, cannot train controller")
            child_model.connect_controller(None)
            controller_ops = None


        print("read checkpoint files...")


        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            thread = tf.train.start_queue_runners(sess=sess,coord=coord)

            saver = tf.train.Saver()

            sess.run(tf.global_variables_initializer())

            # ckpt_state = tf.train.get_checkpoint_state("outputs/search_result")
            
            saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, FLAGS.checkpoint))
            
            
            pred_img_op = child_model.test_preds
            output_dir = child_model.output_dir
            for i in range(4):
                num_batches = meta_data[test_set[i]]["total_data_num"]
                test_img = images[test_set[i]]
                test_label = labels[test_set[i]]
                total_PSNR = 0


                for batch_id in range(num_batches):
                    h_i, w_i, c_i = test_img[batch_id].shape
                    h_o, w_o, c_o = test_label[batch_id].shape
                    inp_test = test_img[batch_id] - bgr_mean

                    test_data = np.reshape(inp_test, [1, h_i, w_i, c_i])
                    pred_img = sess.run(pred_img_op, feed_dict={child_model.x_test: test_data})
                    pred_img = np.reshape(pred_img, [h_o, w_o, c_o]) + bgr_mean
                    pred_img = (np.round(np.clip(pred_img * 255., 0., 255.)) / 255).astype(np.float32)
                    input_img = test_img[batch_id]
                    label_img = test_label[batch_id]
                    pred_img = pred_img[2:-2, 2:-2, :]
                    input_img = input_img[2:-2, 2:-2, :]
                    label_img = label_img[2:-2, 2:-2, :]

                    pred_img_y = bgr2y(pred_img)
                    label_img_y = bgr2y(label_img)
                    # cv2.imshow('pred',pred_img*255)
                    # cv2.imshow('label', label_img*255)
                    result_path = os.path.join(output_dir, "result_img")
                    if not os.path.isdir(result_path):
                        print("Path {} does not exist. Creating.".format(result_path))
                        os.makedirs(result_path)
                    if not os.path.isdir(os.path.join(output_dir, test_set[i])):
                        print("Path {} does not exist. Creating.".format(os.path.join(output_dir, test_set[i])))
                        os.makedirs(os.path.join(output_dir, test_set[i]))
    
                    cv2.imwrite(
                        os.path.join(output_dir, test_set[i], "{}.png".format(batch_id)),
                        pred_img * 255)
                    cv2.imwrite(os.path.join(output_dir, test_set[i], "{}_label.png".format(batch_id)),
                                label_img * 255)
                    cv2.imwrite(os.path.join(output_dir, test_set[i], "{}_input.png".format(batch_id)),
                                input_img * 255)

                    # cv2.waitKey()
                    PSNR = calc_psnr(pred_img_y, label_img_y)
                    total_PSNR += PSNR
                    print("image_{}'s PSNR = {}".format(batch_id, PSNR))
                total_exp = num_batches
                psnr = total_PSNR / total_exp
                print("test_PSNR of {}={:<6.4f}".format(test_set[i],psnr))


def main(_):
    print("-" * 80)


    if not os.path.isdir(FLAGS.checkpoint_dir):
        print("Path {} does not exist. Can't find checkpoint.".format(FLAGS.checkpoint_dir))
    else:
        print("checkpoint exists. at {}".format(FLAGS.checkpoint_dir))
        print("-" * 80)
        utils.print_user_flags()
        evaluate()


if __name__ == "__main__":
    tf.app.run()
