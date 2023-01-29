import torch
import torch.nn as nn
import threading
from torch._utils import ExceptionWrapper
import logging

def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None

def parallel_apply(fct, model, inputs, device_ids):
    modules = nn.parallel.replicate(model, device_ids)
    assert len(modules) == len(inputs)
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input):
        torch.set_grad_enabled(grad_enabled)
        device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = fct(module, *input)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=(i, module, input))
                   for i, (module, input) in enumerate(zip(modules, inputs))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs

logger_initialized = {}
def get_logger(filename=None):
    if "seg" in logger_initialized:
        return logger_initialized["seg"]
    else:
        assert filename is not None
        try:
            from mmcv.utils import get_logger
            from termcolor import colored
            fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
            color_fmt = colored('[%(asctime)s %(name)s]', 'green') \
                        + colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'
            logger = get_logger("seg", log_file=filename, log_level=logging.INFO, file_mode='a')
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))

                if isinstance(handler, logging.FileHandler):
                    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        except Exception as e:
            logger = logging.getLogger("seg")
            logger.setLevel(logging.DEBUG)
            logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                                datefmt='%m/%d/%Y %H:%M:%S',
                                level=logging.INFO)
            if filename is not None:
                handler = logging.FileHandler(filename)
                handler.setLevel(logging.DEBUG)
                handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
                logging.getLogger().addHandler(handler)
        logger_initialized["seg"] = logger
    return logger