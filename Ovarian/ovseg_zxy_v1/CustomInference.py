import logging

import torch
import numpy as np
import pickle
import json

from aiaa.configs.modelconfig import ModelConfig
from aiaa.inference.inference import Inference
from aiaa.utils.class_utils import instantiate_class

from ovseg_zxy.networks.UNet import UNet
from ovseg_zxy.utils.io import read_nii
from ovseg_zxy.utils.torch_np_utils import check_type
from ovseg_zxy.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from ovseg_zxy.prediction.SlidingWindowPrediction import SlidingWindowPrediction
from ovseg_zxy.postprocessing.SegmentationPostprocessing import SegmentationPostprocessing

class CustomInference(Inference):

    def __init__(
        self,
        is_batched_data=False,
        network= None,
        device='cuda'
    ):
        print('Init: network = %s, device = %s, batched = %s'%(network, device, is_batched_data))
        self.network = None
        self.device = device
        self.is_batched_data = is_batched_data

        self.model = None
        print('End of init')
        
    def inference(self, name, data, config: ModelConfig, triton_config):
        print('inference now')
        logger = logging.getLogger(__name__)
        logger.info('Run CustomInference for: {}'.format(name))

        #if self.model is None:
        #    self._init_context(config)

        input_key = config.get_inference_input()
        output_key = config.get_inference_output()
        logger.info('Input Key: {}; Output Key: {}'.format(input_key, output_key))

        #print(data)
        #metadata = data['image_meta_dict']
        #print(metadata)

        # Create data tpl with necessary elements
        data_tpl = {}
        raw_image_file = data["image_original"]
        im, spacing, had_z_first = read_nii(raw_image_file)
        data_tpl['image'] = im
        data_tpl['spacing'] = spacing
        data_tpl['had_z_first'] = had_z_first
        data_tpl['raw_image_file'] = raw_image_file
        print(spacing)
        print(data_tpl)
        print(im.shape)

        # Read model parameters
        model_params_pre_file = '/aiaa_workspace/aiaa-1/lib/ovseg_zxy/parameters/model_parameters_preprocessing.npy'
        #with open(model_params_file, 'rb') as file:
        #    model_params = pickle.load(file)
        #json.load(open(model_params_file))
        model_params_pre = np.load(model_params_pre_file, allow_pickle=True)
        print(model_params_pre)
        #print(model_params['preprocessing'])

        image_key='image'
        pred_fps_key='pred_fps'

        im = data_tpl[image_key]
        is_np,  _ = check_type(im)
        is_cascade = pred_fps_key in data_tpl
        print('Is cascade? %s' %is_cascade) # Cascade won't be implemented for now
        print('Is numpy? %s' %is_np) # Is numpy, we will be getting this anyway, unnecessary
        if len(im.shape) == 3:
            print('Make new axis')
            im = im[np.newaxis] if is_np else im.unsqueeze(0)

        print('*** PREPROCESSING ***')    
        preprocessing = SegmentationPreprocessing(**model_params_pre.item())
        im = preprocessing(data_tpl, preprocess_only_im=True)
        print(im)

        print('*** RUNNING THE MODEL ***')
        print('the fun starts...')

        model_params_network_file = '/aiaa_workspace/aiaa-1/lib/ovseg_zxy/parameters/model_parameters_network.npy'
        model_params_network = np.load(model_params_network_file, allow_pickle=True)
        print(model_params_network)

        network = UNet(**model_params_network.item()).to(self.device)
        path_to_weights = '/aiaa_workspace/aiaa-1/lib/ovseg_zxy/networks/network_weights/v1/network_weights'
        network.load_state_dict(torch.load(path_to_weights,
                                         map_location=torch.device(self.device)))

        model_params_data_file = '/aiaa_workspace/aiaa-1/lib/ovseg_zxy/parameters/model_parameters_data.npy'
        model_params_data = np.load(model_params_data_file, allow_pickle=True)
        prediction_params = {'network': network,
                             'patch_size': model_params_data.item()['trn_dl_params']['patch_size']}
        prediction = SlidingWindowPrediction(**prediction_params)
        pred = prediction(im)
        
        pred_key = 'pred'
        data_tpl[pred_key] = pred

        print('*** POSTPROCESSING ***')
        postprocessing_params = {}
        postprocessing = SegmentationPostprocessing(**postprocessing_params)
        postprocessing.postprocess_data_tpl(data_tpl, pred_key)
        arr = data_tpl[pred_key]
        if 'had_z_first' in data_tpl:
            if not data_tpl['had_z_first']:
                arr = np.stack([arr[z] for z in range(arr.shape[0])], -1)
        data_tpl[pred_key] = arr
        
        print('*** SAVING AND FINISHING INFERENCE ***')
        data.update({output_key: data_tpl[pred_key]})
        logger.info('There are {} non-zero values in the prediction.'.format(np.count_nonzero(data_tpl[pred_key])))
        return data
            
    def _init_context(self, config: ModelConfig):
        logger = logging.getLogger(__name__)
        print('_init_context')

    def _simple_inference(self, inputs):
        pass

    def close(self):
        if self.model and hasattr(self.model, 'close'):
            self.model.close()
        self.model = None
