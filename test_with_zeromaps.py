import os
import cv2
import os.path as osp
import logging
import argparse
import options.options as option
import utils.util as util
from utils.util import opt_get
from data import create_dataloader
from models import create_model
from data.LoL_dataset import load_dataset
def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    # opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    model_path = opt_get(opt, ['model_path'], None)
    model.load_network(load_path=model_path, network=model.netG)
    return model, opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", default="confs/LOLv2-pc.yml")
    args = parser.parse_args()
    conf_path = args.opt
    model, opt = load_model(conf_path)
    model.netG = model.netG.cuda()
    save_imgs = True
    save_folder = 'zero_map_results/{}'.format(opt['name'])
    util.mkdirs(save_folder)
    dataset_cls = load_dataset
    for phase, dataset_opt in opt['datasets'].items():
        val_set = dataset_cls(opt=dataset_opt, train=False, all_opt=opt)
        val_loader = create_dataloader(False, val_set, dataset_opt, opt, None)
    idx = 0
    for val_data in val_loader:
        idx += 1
        img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
        model.feed_data(val_data)
        model.test()
        visuals = model.get_current_visuals()
        rlt_img = util.tensor2img(visuals['NORMAL'])  # uint8
        if save_imgs:
            try:
                print(idx)
                cv2.imwrite(osp.join(save_folder, '{}.png'.format(img_name)), rlt_img)
            except Exception as e:
                print(e)
                import ipdb
                ipdb.set_trace()
if __name__ == '__main__':
    main()
