import torch
import numpy as np
import os
from myprog import myimtocol, dither
import copy


def build_test_loader(config, inputs):
    inputs = np.reshape(inputs, (1,) + inputs.shape)
    inputs = inputs.transpose(1, 0, 2, 3)
    inputs = torch.from_numpy(inputs)

    dataset_test = torch.utils.data.TensorDataset(inputs, inputs)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )
    return data_loader_test


def build_load_folder(config):
    config.defrost()
    dataset_train, dataset_val = build_fromfolder(config)
    config.freeze()

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    return dataset_train, dataset_val, data_loader_train, data_loader_val


class NewDataFolder(torch.utils.data.Dataset):
    def __init__(self, root_dir, model_type=''):
        self.root_dir = root_dir
        if model_type=='NAFnet':
            self.image_path = root_dir + '/swap'
        else:
            self.image_path = root_dir + '/sample'
        self.label_path = root_dir + '/target'
        image_temp = os.listdir(self.image_path)
        self.image_list = []
        self.label_list = []
        for image in image_temp:
            if '.npy' in image:
                self.image_list.append(image)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        X_train = np.load(self.image_path + '/' + self.image_list[index]).astype(np.float32)  ### C H W
        Y_train = np.load(self.label_path + '/' + self.image_list[index]).astype(np.float32)  ### H W
        return X_train, Y_train, self.image_list[index]


def build_fromfolder(config):
    dataset_train = NewDataFolder(config.DATA.DATA_PATH,config.MODEL.TYPE)
    dataset_val = NewDataFolder(config.DATA.TEST_PATH,config.MODEL.TYPE)
    return dataset_train, dataset_val


def patch_single(config, inputs, targets=None):
    inputs = inputs.numpy()
    outputs = myimtocol(inputs[0, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, config.DATA.ROW, config.DATA.COL,
                        config.DATA.SROW, config.DATA.SCOL, 1)
    B, H, W = outputs.shape
    outputs = outputs.reshape(B, 1, H, W)
    if targets is not None:
        outs = myimtocol(targets[0, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, config.DATA.ROW, config.DATA.COL,
                         config.DATA.SROW, config.DATA.SCOL, 1)
        B, H, W = outs.shape
        outs = outs.reshape(B, 1, H, W)
        Y_traint = torch.from_numpy(outs)
        X_traint = torch.from_numpy(outputs)
        return X_traint, Y_traint
    else:
        X_traint = torch.from_numpy(outputs)
        return X_traint


def patching_test(config, inputs):
    index = torch.zeros((2,), dtype=torch.float32)
    index[0] = 10
    index[1] = 10
    index = torch.cat([index, index], axis=0)
    index = torch.cat([index, index], axis=0)
    C, H, W = inputs.shape
    for i in range(C):
        if i == 0:
            outputs = myimtocol(inputs[i, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, 32, 35, 1)
            B, H, W = outputs.shape
            outputs = outputs.reshape(B, 1, H, W)
        else:
            temp = myimtocol(inputs[i, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, 32, 35, 1)
            B, H, W = temp.shape
            temp = temp.reshape(B, 1, H, W)
            outputs = np.concatenate((outputs, temp), axis=0)
    return torch.from_numpy(outputs), index


def patching(config, inputs, name, targets=None, test=False, denoise=False):
    inputs = inputs.numpy()
    print (inputs.shape)
    flag1 = flag2 = False 
    index = torch.zeros((2,), dtype=torch.float32)
    names = name.split('.')[0]
    [n1,n2] = inputs.shape
    index[1] = int(names[1:])
    if names[0] == 'a':
        index[0] = 5 if test else 0
        H, W, sh, sw = n1, n2, 16, 16
    elif names[0] == 'b':
        index[0] = 6 if test else 1
        H, W, sh, sw = n1, n2, 16, 16

    index = torch.cat([index, index], axis=0)
    index = torch.cat([index, index], axis=0)
    
    newinputs = np.zeros((1, 1, H, W), dtype=np.float32)
    newinputs[0, 0, :, :]  = inputs[0, :, :W]
    inputs = newinputs
    newtargets = np.zeros((1, 1, H, W), dtype=np.float32)
    newtargets[0, 0, :, :] = targets[0, :, :W]
    targets = newtargets

    B, C, H, W = inputs.shape
    for i in range(C):
        if i == 0:
            outputs = myimtocol(inputs[0, i, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
            B, H, W = outputs.shape
            outputs = outputs.reshape(B, 1, H, W)
        else:
            temp = myimtocol(inputs[0, i, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
            B, H, W = temp.shape
            temp = temp.reshape(B, 1, H, W)
            outputs = np.concatenate((outputs, temp), axis=0)

    if targets is not None:
        for i in range(C):
            if i == 0:
                outs = myimtocol(targets[0, i, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
                B, H, W = outs.shape
                outs = outs.reshape(B, 1, H, W)
            else:
                temp = myimtocol(targets[0, i, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
                B, H, W = temp.shape
                temp = temp.reshape(B, 1, H, W)
                outs = np.concatenate((outs, temp), axis=0)
    else:
        X_traint = torch.from_numpy(outputs)
        return X_traint, index
    if denoise:
        if flag1 and not test:
            abspath = os.getcwd()
            delay = np.load(abspath + '/' + config.DATA.DATA_PATH + '/delay/' + name)
            obsers = np.load(abspath + '/' + config.DATA.DATA_PATH + '/sample/' + name)
            obser1, obser2 = obsers[0, :, :], obsers[1, :, :]
            ntemp1 = obser1 - dither(inputs[0, 1, :, :], delay) * np.random.uniform(0.75, 0.91)
            ntemp2 = obser2 - dither(inputs[0, 0, :, :], -delay) * np.random.uniform(0.75, 0.91)
            ntemp1 = myimtocol(ntemp1, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
            ntemp2 = myimtocol(ntemp2, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
            out2 = np.concatenate((ntemp1, ntemp2), axis=0)
            B, H, W = out2.shape
            outputs = out2.reshape(B, 1, H, W)
            outputs = np.concatenate((outputs, outputs), axis=0)
            outs = np.concatenate((outs, outs), axis=0)
        if flag2 and not test:
            outputs = np.concatenate((outputs, outputs + np.random.normal(0, 0.05, outputs.shape).astype(np.float32)),
                                     axis=0)
            outs = np.concatenate((outs, outs), axis=0)
            outputs = outputs[:264, :, :, :]
            outs = outs[:264, :, :, :]
    else:
        if flag1 and not test:
            outputs = np.concatenate((outputs, outputs + np.random.normal(0, 0.05, outputs.shape).astype(np.float32)),
                                     axis=0)
            outputs = outputs[:264, :, :, :]
            outs = np.concatenate((outs, outs), axis=0)
            outs = outs[:264, :, :, :]
        if flag2 and not test:
            outputs2 = copy.deepcopy(outputs)
            for i in range(outputs2.shape[0]):
                outputs2[i, :, :, :] *= np.random.uniform(0.95, 1.05)
            outputs = np.concatenate((outputs, outputs2), axis=0)
            outs = np.concatenate((outs, outs), axis=0)
            outputs = outputs[:264, :, :, :]
            outs = outs[:264, :, :, :]
    B,C,H,W = outs.shape
    if config.DATA.BATCH_SIZE < B:
        random_slice=np.random.choice(B,config.DATA.BATCH_SIZE,replace=False)
        Y_traint = torch.from_numpy(outs[random_slice, :, :, :])
        X_traint = torch.from_numpy(outputs[random_slice, :, :, :])
    else:
        Y_traint = torch.from_numpy(outs[:256, :, :, :])
        X_traint = torch.from_numpy(outputs[:256, :, :, :])
        
    # print(X_traint.shape)
    # 
    return X_traint, Y_traint, index



def patching_nomo(config, inputs, name, targets=None, test=False, denoise=False):
    inputs = inputs.numpy()
    flag1 = flag2 = False 
    index = torch.zeros((2,), dtype=torch.float32)
    names = name.split('.')[0]
    index[1] = int(names[1:])
    if names[0] == 'a':
        index[0] = 5 if test else 0
        H, W, sh, sw = 2500, 760, 64, 64
    elif names[0] == 'b':
        index[0] = 6 if test else 1
        H, W, sh, sw = 2500, 760, 64, 64

    index = torch.cat([index, index], axis=0)
    index = torch.cat([index, index], axis=0)
    
    newinputs = np.zeros((1, 1, H, W), dtype=np.float32)
    newinputs[0, 0, :, :]  = inputs[0, :, :W]
    inputs = newinputs
    newtargets = np.zeros((1, 1, H, W), dtype=np.float32)
    newtargets[0, 0, :, :] = targets[0, :, :W]
    targets = newtargets

    B, C, H, W = inputs.shape
    for i in range(C):
        if i == 0:
            outputs = myimtocol(inputs[0, i, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
            B, H, W = outputs.shape
            outputs = outputs.reshape(B, 1, H, W)
        else:
            temp = myimtocol(inputs[0, i, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
            B, H, W = temp.shape
            temp = temp.reshape(B, 1, H, W)
            outputs = np.concatenate((outputs, temp), axis=0)

    if targets is not None:
        for i in range(C):
            if i == 0:
                outs = myimtocol(targets[0, i, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
                B, H, W = outs.shape
                outs = outs.reshape(B, 1, H, W)
            else:
                temp = myimtocol(targets[0, i, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
                B, H, W = temp.shape
                temp = temp.reshape(B, 1, H, W)
                outs = np.concatenate((outs, temp), axis=0)
    else:
        X_traint = torch.from_numpy(outputs)
        return X_traint, index
    if denoise:
        if flag1 and not test:
            abspath = os.getcwd()
            delay = np.load(abspath + '/' + config.DATA.DATA_PATH + '/delay/' + name)
            obsers = np.load(abspath + '/' + config.DATA.DATA_PATH + '/sample/' + name)
            obser1, obser2 = obsers[0, :, :], obsers[1, :, :]
            ntemp1 = obser1 - dither(inputs[0, 1, :, :], delay) * np.random.uniform(0.75, 0.91)
            ntemp2 = obser2 - dither(inputs[0, 0, :, :], -delay) * np.random.uniform(0.75, 0.91)
            ntemp1 = myimtocol(ntemp1, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
            ntemp2 = myimtocol(ntemp2, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
            out2 = np.concatenate((ntemp1, ntemp2), axis=0)
            B, H, W = out2.shape
            outputs = out2.reshape(B, 1, H, W)
            outputs = np.concatenate((outputs, outputs), axis=0)
            outs = np.concatenate((outs, outs), axis=0)
        if flag2 and not test:
            outputs = np.concatenate((outputs, outputs + np.random.normal(0, 0.05, outputs.shape).astype(np.float32)),
                                     axis=0)
            outs = np.concatenate((outs, outs), axis=0)
            outputs = outputs[:264, :, :, :]
            outs = outs[:264, :, :, :]
    else:
        if flag1 and not test:
            outputs = np.concatenate((outputs, outputs + np.random.normal(0, 0.05, outputs.shape).astype(np.float32)),
                                     axis=0)
            outputs = outputs[:264, :, :, :]
            outs = np.concatenate((outs, outs), axis=0)
            outs = outs[:264, :, :, :]
        if flag2 and not test:
            outputs2 = copy.deepcopy(outputs)
            for i in range(outputs2.shape[0]):
                outputs2[i, :, :, :] *= np.random.uniform(0.95, 1.05)
            outputs = np.concatenate((outputs, outputs2), axis=0)
            outs = np.concatenate((outs, outs), axis=0)
            outputs = outputs[:264, :, :, :]
            outs = outs[:264, :, :, :]
    B,C,H,W = outs.shape
    if config.DATA.BATCH_SIZE < B:
        random_slice=np.random.choice(B,config.DATA.BATCH_SIZE,replace=False)
        Y_traint = torch.from_numpy(outs[random_slice, :, :, :])
        X_traint = torch.from_numpy(outputs[random_slice, :, :, :])
    else:
        Y_traint = torch.from_numpy(outs[:256, :, :, :])
        X_traint = torch.from_numpy(outputs[:256, :, :, :])
        
    
    return X_traint, Y_traint, index