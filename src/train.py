import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import SiameseVanillaDataset
from contrastive_loss import ContrastiveLoss
from model_dispatcher import MODEL_DISPATCHER

parser = argparse.ArgumentParser()
parser.add_argument("--NAME", type = str, default="Random_Test" ,help="To give a name to the experiment. Default is 'Random_Test'")
parser.add_argument("--MODEL", type = str, default="siamese_vanilla", help="Which model architecture to use for training")
parser.add_argument("--EPOCHS", type = int, default=100, help="Number of epochs to train the model for")
parser.add_argument("--TRAIN_BATCH_SIZE", type = int, default=64, help="Batchsize to use for model training")
parser.add_argument("--TEST_BATCH_SIZE", type = int, default=64, help="Batchsize to use for model testing")
parser.add_argument("--TRAIN_DATA_PATH", type = str, help="Path to the training data location")
parser.add_argument("--TEST_DATA_PATH", type = str, help="Path to the testing data location")

options, _ = parser.parse_known_args()

os.chdir('../results/')
os.mkdir(f"{options.NAME}")
os.chdir(f'./{options.NAME}')

def train(optimizer, data_loader, train_batch_size, model):
    model.train()
    for i, (pair, match) in enumerate(data_loader):


    optimizer.zero_grad()
    op = model(input)
    loss = ContrastiveLoss.forward(op)

    loss.backward()
    optimizer.step()



def main(parser):
    model = MODEL_DISPATCHER[parser.NAME]
    model.to(parser.DEVICE)

    train_dataset = SiameseVanillaDataset(
        sigDataset = parser.TRAIN_DATA_PATH,
        img_height = parser.IMG_HEIGHT,
        img_width = parser.IMG_WIDTH,
        mean = parser.DATA_MEAN,
        std = parser.DATA_STD
    )

    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = parser.TRAIN_BATCH_SIZE,
        shuffle = True,
        num_workers=4,
        drop_last=True)

    test_dataset = SiameseVanillaDataset(
        sigDataset = parser.TEST_DATA_PATH,
        img_height = parser.IMG_HEIGHT,
        img_width = parser.IMG_WIDTH,
        mean = parser.DATA_MEAN,
        std = parser.DATA_STD
    )
    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = parser.TEST_BATCH_SIZE,
        shuffle = True,
        num_workers=4,
        drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=max, patience=5, factor = 0.3, verbose=True)

    if torch.cuda.device_count > 1:
        model = nn.DataParallel()

    for epoch in tqdm(range(parser.EPOCHS)):
        pass
        train(optimizer, data_loader, parser.TRAIN_BATCH_SIZE, model)
        score = evaluate()
        scheduler.step(score)


if __name__ == "__main__":
    main(parser=options)