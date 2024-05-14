from scipy.ndimage.filters import gaussian_filter
from torchvision import transforms
from CSRNet import CSRNet
from tqdm import tqdm
import argparse
import os.path
import torch
import cv2


# 初始化设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 如果爆显存就把下面这行注释去掉
# device = 'cpu'
t_device = torch.device(device)
# 初始化模型
net = CSRNet().to(device)
info = torch.load('model_best_final.pth', map_location=t_device)
net.load_state_dict(info['state_dict'])
args = ...


def create_args():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='input file name(.png .jpg .mp4)')
    parser.add_argument('-o', '--output', default='output',
                        type=str, help='output file name')
    parser.add_argument('-i', '--interval', default=2,
                        type=int, help='frame interval of output file')
    parser.add_argument('--raw', action='store_true', default=False,
                        help='Whether to output raw data')
    parser.add_argument('--nofont', action='store_true', default=False,
                        help='Whether to put the text on the output image')
    parser.add_argument('--only', action='store_true', default=False,
                        help='Whether to output density map only')

    args = parser.parse_args()


def main():
    create_args()
    kwargs = {'raw': args.raw, 'only': args.only, 'nofont': args.nofont}
    filename = args.filename
    filetype = filename.split('.')[-1]
    if not os.path.isfile(filename):
        print(f'文件{filename}不存在')
        exit(-1)
    output_filename = args.output + (('.' + filetype) if args.output.split('.')[-1] != filetype else '')
    if filetype in ['png', 'jpg']:
        process_image(filename, output_filename, **kwargs)
    elif filetype in ['mp4']:
        process_video(filename, output_filename, args.interval, **kwargs)
    else:
        print(f'不支持.{filetype}类型文件，请输入(.png .jpg .mp4)类型文件')
        exit(-1)


def count_crowd_image(img, raw=False, only=False, nofont=False):
    """标记图片人数"""
    with torch.no_grad():
        # 标准化图片(w, h, c) -> (1, c, w, h)
        temp = img.copy()
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=(img[0].mean(), img[1].mean(), img[2].mean()),
                                   std=(img[0].std(), img[1].std(), img[2].std()))(img).unsqueeze(0).to(device)
        # 生成密度信息(1, c, w, h) -> (1, 1, w//8, h//8)
        output = net(img)
        # 人群计数
        n = int(output.sum() + 0.5)
        # 根据密度信息生成密度图(1, 1, w//8, h//8) -> (3, w//8, h//8)
        output = output[0, 0]
        if raw:
            return n, (output * 255).cpu().numpy()
        # -高斯模糊
        output = torch.Tensor(gaussian_filter(output.cpu(), 1))
        # -数据归一
        output = (output - output.min()) / (output.max() - output.min())
        # output = output / 0.15
        # -颜色转换
        h = 240 * (1 - output)
        s = torch.zero_(output) + 1
        v = torch.zero_(output) + 180
        output = torch.stack((h, s, v), dim=0)
        output = cv2.cvtColor(output.numpy().transpose(1, 2, 0), cv2.COLOR_HSV2BGR)
        # 将图片缩放为img大小
        output = cv2.resize(output, (img.shape[3], img.shape[2]), interpolation=cv2.INTER_LINEAR)
        if only:
            return n, output
        # 合并图片
        output = cv2.addWeighted(temp, 0.7, output, 0.9, 0, dtype=cv2.CV_8U)
        if nofont:
            return n, output
        # 添加文字
        cv2.putText(output, f'Count: {n}', (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (30, 30, 30), 2)
    return n, output


def process_image(filename, output_filename, **kwargs):
    """处理照片"""
    img = cv2.imread(filename)
    n, op = count_crowd_image(img, **kwargs)
    cv2.imwrite(output_filename, op)
    print('预测人数:', n)


def process_video(filename, output_filename, interval, **kwargs):
    """处理视频"""
    # 打开视频
    video = cv2.VideoCapture(filename)
    writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'),
                             video.get(cv2.CAP_PROP_FPS), (int(video.get(3)), int(video.get(4))))
    ret = video.isOpened()
    if not ret:
        print(f'无法打开视频{filename}')
        exit(-1)
    total = video.get(cv2.CAP_PROP_FRAME_COUNT)
    bar = tqdm(total=total)
    temp_f = None
    with bar:
        while True:
            # 读取帧数据
            ret, frame = video.read()
            if not ret:
                break
            if bar.n % interval == 0:
                n, temp_f, = count_crowd_image(frame, **kwargs)
            # 写入到新视频
            writer.write(temp_f)
            bar.update(1)
            yield bar.n / bar.total
    video.release()
    writer.release()


if __name__ == '__main__':
    main()
