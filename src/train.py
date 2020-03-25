import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision
from dataset import SiameseVanillaDataset
from contrastive_loss import ContrastiveLoss
from model_dispatcher import MODEL_DISPATCHER

parser = argparse.ArgumentParser()
parser.add_argument("--NAME", type = str, default="Random_Test" ,help="To give a name to the experiment. Default is 'Random_Test'")
parser.add_argument("--MODEL", type = str, default="siamese_vanilla", help="Which model architecture to use for training")
parser.add_argument("--EPOCHS", type = int, default=100, help="Number of epochs to train the model for")
parser.add_argument("--IMG_HEIGHT", type = int, default=520, help="Default image height")
parser.add_argument("--IMG_WIDTH", type = int, default=200, help="Default image width")
parser.add_argument("--DATA_MEAN", type = int, default=0.5, help="Default image width")
parser.add_argument("--DATA_STD", type = int, default=0.5, help="Default image width")

parser.add_argument("--TRAIN_BATCH_SIZE", type = int, default=4, help="Batchsize to use for model training")
parser.add_argument("--TEST_BATCH_SIZE", type = int, default=4, help="Batchsize to use for model testing")
parser.add_argument("--TRAIN_DATA_PATH", default="/home/transpacks/Repos/Siamese-Network/input/", type = str, help="Path to the training data location")
parser.add_argument("--TEST_DATA_PATH", default="/home/transpacks/Repos/Siamese-Network/test", type = str, help="Path to the testing data location")
parser.add_argument("--DEVICE", type = str, default="cuda:0", help="GPU?CPU, default is GPU- cuda:0")
options, _ = parser.parse_known_args()

os.chdir('../results/')
#os.mkdir(f"{options.NAME}")
os.chdir(f'./{options.NAME}')

def train(optimizer, data_loader, train_batch_size, model, options):
    model.train()
    val_loss = 0
    counter = 0
    for i, (ip_1, ip_2, match) in enumerate(data_loader):
        counter += 1
        ip_1 = ip_1.to(options.DEVICE)
        ip_2 = ip_2.to(options.DEVICE)
        match = match.to(options.DEVICE)

        optimizer.zero_grad()

        op_1, op_2 = model(ip_1, ip_2)
        loss = ContrastiveLoss().forward(op_1, op_2, match)

        loss.backward()
        optimizer.step()
        val_loss += loss
    return (val_loss/counter)

def evaluate(optimizer, data_loader, test_batch_size, model, options):
    model.eval()
    val_loss = 0
    counter = 0
    with torch.no_grad():
        for i, (ip_1, ip_2, match) in enumerate(data_loader):
            counter += 1
            ip_1 = ip_1.to(options.DEVICE)
            ip_2 = ip_2.to(options.DEVICE)
            match = match.to(options.DEVICE)

            op_1, op_2 = model(ip_1, ip_2)
            loss = ContrastiveLoss().forward(op_1, op_2, match)
            val_loss += loss
    return (val_loss/counter)


def main(parser):
    model = MODEL_DISPATCHER["siamese_vanilla"]
    print(parser.DEVICE)
    model.to(parser.DEVICE)

    imgFD_train = torchvision.datasets.ImageFolder(root=parser.TRAIN_DATA_PATH)
    imgFD_test = torchvision.datasets.ImageFolder(root=parser.TEST_DATA_PATH)

    train_dataset = SiameseVanillaDataset(
        imageFolderDataset = imgFD_train,
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
        imageFolderDataset = imgFD_test,
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor = 0.3, verbose=True)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    for epoch in tqdm(range(parser.EPOCHS), total=101):
        score = train(optimizer, train_loader, parser.TRAIN_BATCH_SIZE, model, options)
        #score = evaluate(optimizer, test_loader, parser.TEST_BATCH_SIZE, model, options)
        print("Current Loss",score)
        scheduler.step(score)
        if(epoch%5==0):
            torch.save(model.state_dict(), f"./model_{epoch}.bin")


if __name__ == "__main__":
    main(parser=options)
