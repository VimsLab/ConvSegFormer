from DataHandling import *
from torch.nn import functional as F
from ConvSegFormer import *

def _weighted_cross_entropy_loss(preds, edges):
	losses = F.binary_cross_entropy_with_logits(preds.float(), edges.float(),weight=None,reduction='none')

	loss = torch.sum(losses) / b
	return loss

def training(epochs, model, dataloader, cuda, optimizer, scheduler = None, f_name = 'model.pt'):
	model.train()
	loss_before = 0.0
	for epoch in range(epochs):
		loss_end = 0.0
		count = 0
		iou = 0.0
		for data in dataloader:
			model.zero_grad()
			images = data[0].to(device = cuda)
			contours = data[1].to(device = cuda)
			output = model(images)
			loss = 0.0
			for i in output:
				h, w = i.size()[2:]
				if h == 512:
					loss += _weighted_cross_entropy_loss(i, contours)
				else:
					loss += _weighted_cross_entropy_loss(nn.Upsample(scale_factor=512//h, mode='bilinear', align_corners=False)(i), contours)
			loss.backward()
			optimizer.step()
			loss_end += loss.mean().item()
			count += 1
		if scheduler is not None:
			scheduler.step()
		loss_before = loss_end/count
		print('[%d/%d]Loss: %.2f' % (epoch, epochs, loss_end/count), flush = True)
		if (epoch % 10 != 0):
			continue

		torch.save({
			'epoch': epoch,
			'model_state_dict': model.module.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': loss_end/count
		}, f = f_name + str(epoch) + '.pt')

	return model


def main(train = False, lr = 0.001, epochs = 30, f_name = 'checkpoints/model.pt', device_list = None, device = 0, batch = 0):
	cuda = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")

	model = ConvSegFormer(deep_supervision = True) ## Larger model. For smaller model, change the kernel_size parameter in ConvSegFormer.py at lines 69 and 82 to '1'.

	if device_list is not None:
		model = nn.DataParallel(model, device_ids = device_list)
	else:
		print(device, flush = True)
	model.to(cuda)
	print(cuda, flush = True)
	dim = 512
	train_set = GPRIAugDataset()
	print(len(train_set), flush = True)
	train_dataloader = DataLoader(train_set, batch_size = batch, shuffle = True, num_workers = 8)
	optimizer = torch.optim.Adam(model.parameters(), lr = lr)
	epoch = 0
	
  	print('No Pre-Training', flush = True)
  	start = time.time()

	if train:
			model = training(abs(epochs - epoch), model, train_dataloader, cuda, optimizer, scheduler, f_name = f_name)
			end = time.time()
			print('Time taken for %d with a batch_size of %d is %.2f hours.' %(epochs, batch, (end - start) / (3600)), flush = True)

	lr = float(sys.argv[1])
	n = int(sys.argv[4])
	b = int(sys.argv[5])
	if n > 1:
		main(train = True, lr = lr, epochs = int(sys.argv[2]), f_name = 'checkpoints/' + str(sys.argv[3]), device_list = [i for i in range(n)], batch = n * b)
	else:
		main(train = True, lr = lr, epochs = int(sys.argv[2]), f_name = 'checkpoints/' + str(sys.argv[3]), device = n, batch = b)
