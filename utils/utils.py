import os
import onnx
from onnx_opcounter import calculate_params
import torch
from onnxsim import simplify
# from google.colab import files
import cifar10
import matplotlib.pyplot as plt
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType, QuantFormat, QuantizationMode, CalibrationMethod
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader, Dataset
import argparse
from tqdm import tqdm
from PIL import Image
import glob
import onnxruntime 
from onnxruntime import quantization
from tqdm import tqdm
import time


class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

def set_seed():
  seed_value= 42
  
  # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
  import os
  os.environ['PYTHONHASHSEED']=str(seed_value)
  
  # 2. Set `python` built-in pseudo-random generator at a fixed value
  import random
  random.seed(seed_value)
  
  # 3. Set `numpy` pseudo-random generator at a fixed value
  import numpy as np
  np.random.seed(seed_value)
  
  # 4. Set `pytorch` pseudo-random generator at a fixed value
  import torch
  torch.manual_seed(seed_value)
  
  torch.cuda.manual_seed_all(seed_value)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

class RandomTensorDataset(Dataset):
    def __init__(self, size, shape=(3, 224, 224)):
        self.size = size
        self.shape = shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate a random tensor and a random label (10 classes)
        data = torch.rand(self.shape)
        label = torch.randint(0, 10, (1,)).item()
        return data, label

def prepare_test_data(random=False):
  set_seed()
  print('==> Preparing data..')

  transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  if random:
    print('Random Dataset')
    testset = RandomTensorDataset(size=dataset_size)
  else:
    print('CIFAR10 Dataset')
    testset = torchvision.datasets.CIFAR10(
      root='./data', train=False, download=True, transform=transform_test)
  
  quant_size = 100
  test_size = len(testset) - quant_size
  test_ds, quant_ds = random_split(testset, [test_size, quant_size])

  print('Len of data ', len(test_ds), len(quant_ds))

  testloader = torch.utils.data.DataLoader(
      test_ds, batch_size=100, shuffle=False, num_workers=2)

  # quantloader = torch.utils.data.DataLoader(
  #     quant_ds, batch_size=100, shuffle=False, num_workers=2)

  return testloader, quant_ds

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)

def get_acc(model_path, data, device = 'cuda'):

  # print(len(data),)
  set_seed()
  def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

  ort_provider = ['CPUExecutionProvider']
  if device == 'cuda' and torch.cuda.is_available():
    ort_provider = ['CUDAExecutionProvider']
  print('Ort_provider - ', ort_provider)
  ort_sess = onnxruntime.InferenceSession(model_path, providers=ort_provider)

  correct_pt = 0
  correct_onnx = 0
  tot_abs_error = 0
  bs = 0

  for img_batch, label_batch in data:
    if bs == 0:
      bs = len(img_batch)
    # print(bs)
    ort_inputs = {ort_sess.get_inputs()[0].name: to_numpy(img_batch)}
    ort_outs = ort_sess.run(None, ort_inputs)[0]

    ort_preds = np.argmax(ort_outs, axis=1)
    correct_onnx += np.sum(np.equal(ort_preds, to_numpy(label_batch)))

  print(f"ONNX top-1 acc = {(100.0 * correct_onnx/(len(data)*bs))} with {correct_onnx} correct samples\n")
  return (100.0 * correct_onnx/(len(data)*bs))

def simplify_model(model_path, save = False):
  """
    Simplify a pre-trained ONNX model using onnx-simplifier.

    Parameters:
        model_path (str): The path to the pre-trained ONNX model.
        save (bool, optional): Flag to download the simplified model after saving. Default is False.

    Returns:
        str: The path to the simplified ONNX model.
  """

  # load your predefined ONNX model
  model = onnx.load(model_path)

  # convert model
  model_simp, check = simplify(model)

  assert check, "Simplified ONNX model could not be validated"

  new_path = model_path.split('.')[0] + '_sim.onnx'
  onnx.save(model_simp, new_path)
  # if save:
  #   files.download(new_path)
  return new_path

def get_onnx_model_info(model_path):
    """
    Get information about an ONNX model, including the number of parameters and model size.

    Parameters:
        model_path (str): The path to the ONNX model.

    Returns:
        dict: A dictionary containing information about the ONNX model.
    """
    print(bcolors.OKBLUE, 'Model -', model_path + bcolors.ENDC)
    ans = {}
    #try:
      # Check if the model file exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"The file '{model_path}' does not exist.")
    # Get the file size in bytes
    model_size_bytes = os.path.getsize(model_path)
    # Load the ONNX model
    onnx_model = onnx.load(model_path)
    # Calculate the number of parameters in the model
    params = calculate_params(onnx_model)
    # Convert the size to a human-readable format
    if model_size_bytes < 1024:
        model_size = f"{model_size_bytes} bytes"
    elif model_size_bytes < 1024 ** 2:
        model_size = f"{model_size_bytes / 1024:.2f} KB"
    elif model_size_bytes < 1024 ** 3:
        model_size = f"{model_size_bytes / (1024 ** 2):.2f} MB"
    else:
        model_size = f"{model_size_bytes / (1024 ** 3):.2f} GB"
    ans =  {
        "params": params,
        "model_size": model_size
    }
    print(f"Number of parameters: {ans['params']}")
    print(f"Model size: {ans['model_size']}")
    return ans

def plot_bar_chart_with_values(x_values, y_values, x_label, y_label, title, legend_labels=None):
    """
    Plot a bar chart with values annotated.

    Parameters:
        x_values (list): X-axis values.
        y_values (list): Y-axis values.
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
        title (str): Title of the graph.
        legend_labels (list, optional): Labels for the legend.

    Returns:
        None
    """
    num_bars = len(x_values)
    bar_width = 0.35
    index = np.arange(num_bars)

    plt.figure(figsize=(8, 6))

    # Plot the data
    if legend_labels:
        for i, y_values_set in enumerate(y_values):
            plt.bar(index + i * bar_width, y_values_set, bar_width, label=legend_labels[i])
            # Annotate values
            for j, value in enumerate(y_values_set):
                plt.annotate(str(value), (index[j] + i * bar_width, value), ha='center', va='bottom')
        plt.legend()
    else:
        plt.bar(index, y_values, bar_width)
        # Annotate values
        for i, value in enumerate(y_values):
            plt.annotate(str(value), (index[i], value), ha='center', va='bottom')

    # Set labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(index + bar_width / 2, x_values)

    # Show the plot
    plt.show()

def export_model(model, model_path, bs = 1, dynamic = False, save = False, sz=32):
  """
    Export a PyTorch model to the ONNX format.

    Parameters:
        model (torch.nn.Module): The PyTorch model to be exported.
        model_path (str): The path where the ONNX model will be saved.
        bs (int, optional): Batch size for model inference. Default is 1.
        dynamic (bool, optional): Flag indicating dynamic axes for variable length inputs. Default is False.
        save (bool, optional): Flag to download the exported model after saving. Default is False.

    Returns:
        None
  """
  model = model.to('cpu')
  model.eval()
  # Input to the model
  batch_size = bs
  x = torch.randn(batch_size, 3, sz, sz, requires_grad=False)
  with torch.no_grad():
    torch_out = model(x)
  # Export the model
  if dynamic:
    torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  model_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}}
                  )
  else:
        torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  model_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                  #               'output' : {0 : 'batch_size'}}
                  )
def benchmark(model_path, bs=100, bs_divide=True, runs=100, device='both', sz=32):
    """
    Функция benchmark используется для измерения времени выполнения инференса модели ONNX на CPU и/или GPU.

    Аргументы:
    - model_path (str): Путь к модели в формате ONNX.
    - bs (int): Размер батча, который будет использоваться для тестирования (по умолчанию 100).
    - bs_divide (bool): Зарезервировано для будущих возможностей, пока не используется.
    - runs (int): Количество прогонов для измерения среднего времени выполнения (по умолчанию 100).
    - device (str): Устройство для тестирования ('cpu', 'gpu', или 'both' для тестирования на обоих).

    Логика:
    1. Устанавливается фиксированный seed для воспроизводимости результатов.
    2. В зависимости от устройства (device):
       - Создается сессия InferenceSession для CPU или GPU с соответствующим провайдером ONNX Runtime.
       - Генерируются входные данные в виде массива numpy с форматом (batch_size, 3, 32, 32) и типом float32.
       - Выполняется разогрев модели (warming up), чтобы исключить влияние первого прогона на результаты.
       - Производится измерение времени выполнения инференса с использованием цикла `for` и фиксируется время выполнения каждого прогона.
    3. Рассчитывается среднее время выполнения для всех прогонов и выводится результат в миллисекундах.

    Пример вывода:
    - Для CPU:
      "CPU Avg: 12.34ms, per 1 img: 0.12ms"
    - Для GPU:
      "GPU Avg: 1.23ms, per 1 img: 0.01ms"

    Особенности:
    - Используются tqdm для отображения прогресса выполнения тестов.
    - Отдельное измерение времени выполнения инференса для каждого устройства.
    - Печать результатов в формате, удобном для анализа производительности.

    Пояснения по коду:
    - `onnxruntime.InferenceSession`: Создает сессию для выполнения инференса с использованием ONNX Runtime.
    - `ortvalue_from_numpy`: Конвертирует входной numpy массив в формат OrtValue, поддерживаемый ONNX Runtime.
    - `time.perf_counter`: Используется для точного измерения времени выполнения.
    - `tqdm`: Добавляет визуальное представление прогресса в циклах.

    Выводы по результатам теста помогут оценить производительность модели на разных устройствах (CPU и GPU).
    """
    set_seed()
    print(bcolors.OKBLUE, f"Model - {model_path}", bcolors.ENDC)
    if device == 'cpu' or device == 'both':
      # CPU benchmark
      ort_provider = ['CPUExecutionProvider']
      session_cpu = onnxruntime.InferenceSession(model_path, providers=ort_provider)
      input_name_cpu = session_cpu.get_inputs()[0].name

      total_cpu = 0.0
      input_data_cpu = np.zeros((bs, 3, sz, sz), np.float32)
      X_ortvalue_cpu = onnxruntime.OrtValue.ortvalue_from_numpy(input_data_cpu, 'cpu')

      # Warming up
      _ = session_cpu.run([], {input_name_cpu: X_ortvalue_cpu})

      for _ in tqdm(range(runs), desc="CPU Benchmark"):
          start_cpu = time.perf_counter()
          _ = session_cpu.run([], {input_name_cpu: X_ortvalue_cpu})
          end_cpu = (time.perf_counter() - start_cpu) * 1000
          total_cpu += end_cpu

      total_cpu /= runs

      print(f"CPU Avg: {total_cpu:.2f}ms, per 1 img: {total_cpu / bs:.2f}ms\n")
    # print()
    if device == 'gpu' or device == 'both':
      # GPU benchmark
      ort_provider_gpu = ['CUDAExecutionProvider']
      session_gpu = onnxruntime.InferenceSession(model_path, providers=ort_provider_gpu)
      input_name_gpu = session_gpu.get_inputs()[0].name

      total_gpu = 0.0
      input_data_gpu = np.zeros((bs, 3, sz, sz), np.float32)
      X_ortvalue_gpu = onnxruntime.OrtValue.ortvalue_from_numpy(input_data_gpu, 'cuda', 0)

      # Warming up
      _ = session_gpu.run([], {input_name_gpu: X_ortvalue_gpu})

      for _ in tqdm(range(runs), desc="GPU Benchmark"):
          start_gpu = time.perf_counter()
          _ = session_gpu.run([], {input_name_gpu: X_ortvalue_gpu})
          end_gpu = (time.perf_counter() - start_gpu) * 1000
          total_gpu += end_gpu

      total_gpu /= runs
      # print()
      # print()
      print(f"GPU Avg: {total_gpu:.2f}, per 1 img: {total_gpu / bs:.2f}ms\n")
      # print()