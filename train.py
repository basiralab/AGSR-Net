import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import *
from model import *
import torch.optim as optim

criterion = nn.MSELoss()


def train(model, subjects_adj, subjects_labels, args):

    bce_loss = nn.BCELoss()
    netD = Discriminator(args)
    print(netD)
    optimizerG = optim.Adam(model.parameters(), lr=args.lr)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)

    all_epochs_loss = []
    for epoch in range(args.epochs):
        with torch.autograd.set_detect_anomaly(True):
            epoch_loss = []
            epoch_error = []
            for lr, hr in zip(subjects_adj, subjects_labels):
                optimizerD.zero_grad()
                optimizerG.zero_grad()

                hr = pad_HR_adj(hr, args.padding)
                lr = torch.from_numpy(lr).type(torch.FloatTensor)
                padded_hr = torch.from_numpy(hr).type(torch.FloatTensor)

                eig_val_hr, U_hr = torch.symeig(
                    padded_hr, eigenvectors=True, upper=True)

                model_outputs, net_outs, start_gcn_outs, layer_outs = model(
                    lr, args.lr_dim, args.hr_dim)

                mse_loss = args.lmbda * criterion(net_outs, start_gcn_outs) + criterion(
                    model.layer.weights, U_hr) + criterion(model_outputs, padded_hr)

                error = criterion(model_outputs, padded_hr)
                real_data = model_outputs.detach()
                fake_data = gaussian_noise_layer(padded_hr, args)

                d_real = netD(real_data)
                d_fake = netD(fake_data)

                dc_loss_real = bce_loss(d_real, torch.ones(args.hr_dim, 1))
                dc_loss_fake = bce_loss(d_fake, torch.zeros(args.hr_dim, 1))
                dc_loss = dc_loss_real + dc_loss_fake

                dc_loss.backward()
                optimizerD.step()

                d_fake = netD(gaussian_noise_layer(padded_hr, args))

                gen_loss = bce_loss(d_fake, torch.ones(args.hr_dim, 1))
                generator_loss = gen_loss + mse_loss
                generator_loss.backward()
                optimizerG.step()

                epoch_loss.append(generator_loss.item())
                epoch_error.append(error.item())

            print("Epoch: ", epoch, "Loss: ", np.mean(epoch_loss),
                  "Error: ", np.mean(epoch_error)*100, "%")
            all_epochs_loss.append(np.mean(epoch_loss))


def test(model, test_adj, test_labels, args):

    g_t = []
    test_error = []
    preds_list = []

    # i = 0

    for lr, hr in zip(test_adj, test_labels):
        all_zeros_lr = not np.any(lr)
        all_zeros_hr = not np.any(hr)
        if all_zeros_lr == False and all_zeros_hr == False:
            lr = torch.from_numpy(lr).type(torch.FloatTensor)
            np.fill_diagonal(hr, 1)
            hr = pad_HR_adj(hr, args.padding)
            hr = torch.from_numpy(hr).type(torch.FloatTensor)
            preds, a, b, c = model(lr, args.lr_dim, args.hr_dim)

            # if i == 0:
            #     print("Hr", hr)
            #     print("Preds  ", preds)
            #     plt.imshow(hr, origin='lower',  extent=[
            #         0, 10000, 0, 10], aspect=1000)
            #     plt.show(block=False)
            #     plt.imshow(preds.detach(), origin='lower',
            #                extent=[0, 10000, 0, 10], aspect=1000)
            #     plt.show(block=False)
            #     plt.imshow(hr - preds.detach(), origin='lower',
            #                extent=[0, 10000, 0, 10], aspect=1000)
            #     plt.show(block=False)

            preds_list.append(preds.flatten().detach().numpy())
            error = criterion(preds, hr)
            g_t.append(hr.flatten())
            print(error.item())
            test_error.append(error.item())
            # i += 1

    print("Test error MSE: ", np.mean(test_error))
    # preds_list = [val for sublist in preds_list for val in sublist]
    # g_t_list = [val for sublist in g_t for val in sublist]
    # binwidth = 0.01
    # bins = np.arange(0, 1 + binwidth, binwidth)
    # plt.hist(preds_list, bins=bins, range=(0, 1),
    #         alpha=0.5, rwidth=0.9, label='predictions')
    # plt.hist(g_t_list, bins=bins, range=(0, 1),
    #         alpha=0.5, rwidth=0.9, label='ground truth')
    # plt.xlim(xmin=0, xmax=1)
    # plt.legend(loc='upper right')
    # plt.title('GSR-UNet with self reconstruction: Histogram')
    # plt.show(block=False)
    # plt.plot(all_epochs_loss)
    # plt.title('GSR-UNet with self reconstruction: Loss')
    # plt.show()
