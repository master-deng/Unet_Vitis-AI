import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from modules import UNet
from tqdm import tqdm
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#torch.cuda.set_per_process_memory_fraction(0.9, device=0)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir',
    default="/path/to/imagenet/",
    help='Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation')
parser.add_argument(
    '--model_dir',
    default="/path/to/trained_model/",
    help='Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: https://download.pytorch.org/models/resnet18-5c106cde.pth'
)
parser.add_argument(
    '--config_file',
    default=None,
    help='quantization configuration file')
parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument(
    '--batch_size',
    default=32,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument('--fast_finetune', 
    dest='fast_finetune',
    action='store_true',
    help='fast finetune model before calibration')
parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')
parser.add_argument('--inspect', 
    dest='inspect',
    action='store_true',
    help='inspect model')

parser.add_argument('--target', 
    dest='target',
    nargs="?",
    const="",
    help='specify target device')

args, _ = parser.parse_known_args()

def load_data(train=True,
              data_dir='./dataset/',
              batch_size=1,
              subset_len=None,
              sample_method='random',
              distributed=False,
              model_name='diffusion',
              **kwargs):

  #prepare data
  # random.seed(12345)
  traindir = data_dir
  valdir = data_dir
  train_sampler = None
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  if model_name == 'inception_v3':
    size = 299
    resize = 299
  else:
    size = 64
    resize = 64
  if train:
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    if subset_len:
      assert subset_len <= len(dataset)
      if sample_method == 'random':
        dataset = torch.utils.data.Subset(
            dataset, random.sample(range(0, len(dataset)), subset_len))
      else:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
    if distributed:
      train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **kwargs)
  else:
    dataset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]))
    if subset_len:
      assert subset_len <= len(dataset)
      if sample_method == 'random':
        dataset = torch.utils.data.Subset(
            dataset, random.sample(range(0, len(dataset)), subset_len))
      else:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, **kwargs)
  return data_loader, train_sampler

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        a = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * a, a

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))



def evaluate(model, val_loader):
    model.eval()
    model = model.to(device)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=64, device=device)
    for epoch in range(100):
        for step, (images, _) in tqdm(enumerate(val_loader), total=len(val_loader)):
                images = images.to(device)
                t = diffusion.sample_timesteps(images.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(images, t)
                predicted_noise = model(x_t, t)
                loss = mse(noise, predicted_noise)
                if step + 1 >= 4:
                   break
    return loss
#
#def evaluate(model, val_loader, loss_fn):
#
#  model.eval()
#  model = model.to(device)
#  top1 = AverageMeter('Acc@1', ':6.2f')
#  top5 = AverageMeter('Acc@5', ':6.2f')
#  total = 0
#  Loss = 0
#  for iteraction, (images, labels) in tqdm(
#      enumerate(val_loader), total=len(val_loader)):
#    images = images.to(device)
#    labels = labels.to(device)
#    #pdb.set_trace()
#    outputs = model(images)
#    loss = loss_fn(outputs, labels)
#    Loss += loss.item()
#    total += images.size(0)
#    acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
#    top1.update(acc1[0], images.size(0))
#    top5.update(acc5[0], images.size(0))
#  return top1.avg, top5.avg, Loss / total

def quantization(title='optimize',
                 model_name='', 
                 file_path=''): 

  data_dir = args.data_dir
  quant_mode = args.quant_mode
  finetune = args.fast_finetune
  deploy = args.deploy
  batch_size = args.batch_size
  subset_len = args.subset_len
  inspect = args.inspect
  config_file = args.config_file
  target = args.target
  if quant_mode != 'test' and deploy:
    deploy = False
    print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
  if deploy and (batch_size != 1 or subset_len != 1):
    print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
    batch_size = 1
    subset_len = 1

  model = UNet().to(device)
  model.load_state_dict(torch.load(file_path))

  input = torch.randn([batch_size, 3, 64, 64], dtype=torch.float32)
  input_t = torch.randn(1, dtype=torch.float32)
  if quant_mode == 'float':
    quant_model = model
    if inspect:
      if not target:
          raise RuntimeError("A target should be specified for inspector.")
      import sys
      from pytorch_nndct.apis import Inspector
      # create inspector
      inspector = Inspector(target)  # by name
      # start to inspect
      inspector.inspect(quant_model, (input,input_t), device=device)
      sys.exit()
      
  else:
    ####################################################################################
    # This function call will create a quantizer object and setup it. 
    # Eager mode model code will be converted to graph model. 
    # Quantization is not done here if it needs calibration.
    quantizer = torch_quantizer(
        quant_mode, model, (input,input_t), device=device, quant_config_file=config_file, target=target)

    # Get the converted model to be quantized.
    quant_model = quantizer.quant_model
    #####################################################################################
 
  # to get loss value after evaluation
  loss_fn = torch.nn.CrossEntropyLoss().to(device)

  val_loader, _ = load_data(
      subset_len=subset_len,
      train=False,
      batch_size=batch_size,
      sample_method='random',
      data_dir=data_dir,
      model_name=model_name)

  # fast finetune model or load finetuned parameter before test
  if finetune == True:
      ft_loader, _ = load_data(
          subset_len=5120,
          train=False,
          batch_size=batch_size,
          sample_method='random',
          data_dir=data_dir,
          model_name=model_name)
      if quant_mode == 'calib':
        quantizer.fast_finetune(evaluate, (quant_model, ft_loader, loss_fn))
      elif quant_mode == 'test':
        quantizer.load_ft_param()
   
  # record  modules float model accuracy
  # add modules float model accuracy here
  acc_org1 = 0.0
  acc_org5 = 0.0
  loss_org = 0.0

  # This function call is to do forward loop for model to be quantized.
  # Quantization calibration will be done after it if quant_mode is 'calib'.
  # Quantization test  will be done after it if quant_mode is 'test'.
  loss_gen = evaluate(quant_model, val_loader)

  # logging accuracy
  # If quant_mode is 'calib', ignore the accuracy log because it is not the final accuracy can be got.
  # Only check the accuray if quant_mode is 'test'.
  print('loss: %g' % (loss_gen))

  # handle quantization result
  if quant_mode == 'calib':
    # Exporting intermediate files will be used when quant_mode is 'test'. This is must.
    quantizer.export_quant_config()
  if deploy:
    quantizer.export_torch_script()
    quantizer.export_onnx_model()
    quantizer.export_xmodel()


if __name__ == '__main__':

  model_name = 'diffusion'
  file_path = os.path.join(args.model_dir, model_name + '.pt')

  feature_test = ' float model evaluation'
  if args.quant_mode != 'float':
    feature_test = ' quantization'
    # force to merge BN with CONV for better quantization accuracy
    args.optimize = 1
    feature_test += ' with optimization'
  else:
    feature_test = ' float model evaluation'
  title = model_name + feature_test

  print("-------- Start {} test ".format(model_name))

  # calibration or evaluation
  quantization(
      title=title,
      model_name=model_name,
      file_path=file_path)

  print("-------- End of {} test ".format(model_name))
