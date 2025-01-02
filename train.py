import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('IRSTD-YOLO.yaml')
    model.train(data='dataset\sirstv2\data.yaml',
                cache=False,
                imgsz=640,
                epochs=500,
                batch=4,
                optimizer='SGD', 
                project='runs/train',
                name='train',
                )
