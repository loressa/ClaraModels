import logging

import torch
import numpy as np
import pickle
import json
import time

from aiaa.configs.modelconfig import ModelConfig
from aiaa.inference.inference import Inference
from aiaa.utils.class_utils import instantiate_class

from ovseg.model.ClaraWrappers import ClaraWrapperOvarian
from ovseg_zxy.utils.io import read_nii

# change this path 
PATH_TO_CLARA_MODELS = '/aiaa_workspace/aiaa-1/lib/ovseg_zxy/clara_models'

class CustomInference(Inference):

    def __init__(
        self,
        is_batched_data=False,
        network= None,
        device='cuda'
    ):
        
        
        self._initialize_models()
        self.path_to_clara_models = PATH_TO_CLARA_MODELS
        
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

        print('***STARTING NETWORK EXECUTION***')

        st = time.perf_counter()
        # this part is new and improved
        pred = ClaraWrapperOvarian(data_tpl,
                                   self.models,
                                   self.path_to_clara_models)
        et = time.perf_counter()
        
        print(f'Execution done after {et-st:.2f} seconds.')
        
        print('*** SAVING AND FINISHING INFERENCE ***')
        data.update({output_key: pred})
        logger.info('There are {} non-zero values in the prediction.'.format(np.count_nonzero(pred)))
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
    
    def _initialize_models(self):
        raise NotImplementedError('Overwrite this method to determine which '
                                  'regions are segmented')

class CustomInferenceOmentumPelivOvarian(CustomInference):
    
    def _initialize_models(self):
        self.models = ['pod_om']
        print('Initialized to segment omental (1) and pelvic/ovarian lesions(9)')

class CustomInferenceAbdominalLesions(CustomInference):
    
    def _initialize_models(self):
        self.models = ['abdominal']
        print('Initialized to segment lesions in the omentum (1), right upper quadrant (2), '
              'left upper quadrant (3), mesentery (5), left paracolic gutter (6), '
              'and right paracolic gutter (7).')

class CustomInferenceLymphNodes(CustomInference):
    
    def _initialize_models(self):
        self.models = ['lymph_nodes']
        print('Initialized to segment lesions in the infrarenal (13), suprerenal (14), '
              'supradiaphramatic (15), and inguinal lymph nodes (17).')

class CustomInferenceAll(CustomInference):
    
    def _initialize_models(self):
        self.models = ['abdominal', 'pod_om', 'lymph_nodes']
        print('Initialized to segment lesions in the infrarenal (13), suprerenal (14), '
              'supradiaphramatic (15), and inguinal lymph nodes (17).')


        