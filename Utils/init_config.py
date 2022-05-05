# @Time   : 2021/6/29
# @Update   : 2021/8/30

import re
import os
import sys
import yaml
import torch
from logging import getLogger

from enum import Enum

class Config(object):
    """ Configurator module that load the defined parameters.

    Configurator module will first load the default parameters from the fixed properties in RecBole and then
    load parameters from the external input.

    External input supports three kind of forms: config file, command line and parameter dictionaries.

    - config file: It's a file that record the parameters to be modified or added. It should be in ``yaml`` format,
      e.g. a config file is 'example.yaml', the content is:

        learning_rate: 0.001

        train_batch_size: 2048

    - command line: It should be in the format as '---learning_rate=0.001'

    - parameter dictionaries: It should be a dict, where the key is parameter name and the value is parameter value,
      e.g. config_dict = {'learning_rate': 0.001}

    Configuration module allows the above three kind of external input format to be used together,
    the priority order is as following:

    command line > parameter dictionaries > config file

    e.g. If we set learning_rate=0.01 in config file, learning_rate=0.02 in command line,
    learning_rate=0.03 in parameter dictionaries.

    Finally the learning_rate is equal to 0.02.
    """

    def __init__(self, model=None, dataset=None):
        """
        Args:
            model (str/AbstractRecommender): the model name or the model class, default is None, if it is None, config
            will search the parameter 'model' from the external input as the model name or model class.
            dataset (str): the dataset name, default is None, if it is None, config will search the parameter 'dataset'
            from the external input as the dataset name.
            config_file_list (list of str): the external config file, it allows multiple config files, default is None.
            config_dict (dict): the external parameter dictionaries, default is None.
        """
        # self._init_parameters_category()
        self.yaml_loader = self._build_yaml_loader()

        # external_config_dict
        # self.file_config_dict = self._load_config_files(config_file_list)
        self.cmd_config_dict = self._load_cmd_line()
        self._merge_external_config_dict()
        # internal_config_dict
        self.model, self.dataset = self._get_model_and_dataset(model, dataset)
        self._load_internal_config_dict(self.model, self.dataset)

        self.final_config_dict = self._get_final_config_dict()
        self._init_device()

    # def _init_parameters_category(self):
    #     self.parameters = dict()
    #     self.parameters['General'] = general_arguments
    #     self.parameters['Training'] = training_arguments
    #     self.parameters['Evaluation'] = evaluation_arguments

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(
                u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X
            ), list(u'-+0123456789.')
        )
        return loader

    def _convert_config_dict(self, config_dict):
        r"""This function convert the str parameters to their original type.

        """
        for key in config_dict:
            param = config_dict[key]
            if not isinstance(param, str):
                continue
            try:
                value = eval(param)
                if not isinstance(value, (str, int, float, list, tuple, dict, bool, Enum)):
                    value = param
            except (NameError, SyntaxError, TypeError):
                if isinstance(param, str):
                    if param.lower() == "true":
                        value = True
                    elif param.lower() == "false":
                        value = False
                    else:
                        value = param
                else:
                    value = param
            config_dict[key] = value
        return config_dict

    # def _load_config_files(self, file_list):
    #     file_config_dict = dict()
    #     if file_list:
    #         for file in file_list:
    #             with open(file, 'r', encoding='utf-8') as f:
    #                 file_config_dict.update(yaml.load(f.read(), Loader=self.yaml_loader))
    #     return file_config_dict

    def _load_cmd_line(self):
        r""" Read parameters from command line and convert it to str.

        """
        cmd_config_dict = dict()
        unrecognized_args = []
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                    unrecognized_args.append(arg)
                    continue
                cmd_arg_name, cmd_arg_value = arg[2:].split("=")
                if cmd_arg_name in cmd_config_dict and cmd_arg_value != cmd_config_dict[cmd_arg_name]:
                    raise SyntaxError("There are duplicate commend arg '%s' with different value." % arg)
                else:
                    cmd_config_dict[cmd_arg_name] = cmd_arg_value
        if len(unrecognized_args) > 0:
            logger = getLogger()
            logger.warning('command line args [{}] will not be used in AdapRanker'.format(' '.join(unrecognized_args)))
        cmd_config_dict = self._convert_config_dict(cmd_config_dict)
        return cmd_config_dict

    def _merge_external_config_dict(self):
        external_config_dict = dict()
        # external_config_dict.update(self.file_config_dict)
        external_config_dict.update(self.cmd_config_dict)
        self.external_config_dict = external_config_dict

    def _get_model_and_dataset(self, model, dataset):
        if model is None:
            try:
                model = self.external_config_dict['model']
            except KeyError:
                raise KeyError(
                    'model need to be specified in at least one of the these ways: '
                    '[model variable, config file, config dict, command line] '
                )
        if not isinstance(model, str):
            final_model = model.__name__
        else:
            final_model = model

        if dataset is None:
            try:
                final_dataset = self.external_config_dict['dataset']
            except KeyError:
                raise KeyError(
                    'dataset need to be specified in at least one of the these ways: '
                    '[dataset variable, config file, config dict, command line] '
                )
        else:
            final_dataset = dataset

        return final_model, final_dataset

    def _update_internal_config_dict(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            config_dict = yaml.load(f.read(), Loader=self.yaml_loader)
            if config_dict is not None:
                self.internal_config_dict.update(config_dict)
        return config_dict

    def _load_internal_config_dict(self, model, dataset):
        current_path = os.path.dirname(os.path.realpath(__file__))
        overall_init_file = os.path.join(current_path, '../config/overall.yaml')
        model_init_file = os.path.join(current_path, '../config/model_config/' + model + '.yaml')
        sample_init_file = os.path.join(current_path, '../config/dataset_config/sample.yaml')
        dataset_init_file = os.path.join(current_path, '../config/dataset_config/' + dataset + '.yaml')

        self.internal_config_dict = dict()
        for file in [overall_init_file, model_init_file, sample_init_file, dataset_init_file]:
            if os.path.isfile(file):
                config_dict = self._update_internal_config_dict(file)
                
    def _get_final_config_dict(self):
        final_config_dict = dict()
        final_config_dict.update(self.internal_config_dict)
        final_config_dict.update(self.external_config_dict)

        final_config_dict['dataset'] = self.dataset
        final_config_dict['model'] = self.model
        final_config_dict['dataset_path'] = final_config_dict['dataset_path']
        final_config_dict['valid_metric_bigger'] = True

        return final_config_dict
        
    def _init_device(self):
        use_gpu = self.final_config_dict['use_gpu']
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict['gpu_id'])
        self.final_config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getitem__(self, item):
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None
    
    def __str__(self):
        args_info = '\n'
        
        args_info += '\n'.join([(("{}") + " =" + (" {}")).format(arg, value)
                                for arg, value in self.final_config_dict.items()
                                ])
        args_info += '\n\n'
        return args_info
    

