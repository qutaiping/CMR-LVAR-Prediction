import argparse
import glob
import os
import sys
import tarfile
import numpy as np
import pandas 
import SimpleITK as sitk
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer.predictor_LAVR_class import (
    ResampledClassificationModel,
    ResampledClassificationPredictor,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test segmask_3d")

    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument(
        "--input_dicom_path", default="main/demo/img",type=str
    )
    parser.add_argument(
            "--input_infar_path", default="main/demo/seg",type=str
    )
    parser.add_argument(
        "--input_flow_path", default="main/demo/flow", type=str
    )
    parser.add_argument("--output_path", default="main/demo/pred", type=str)
    parser.add_argument(
        "--model_path",
        default=glob.glob("main/data/model/*.tar")[0] if len(glob.glob("main/data/model/*.tar")) > 0 else None,
        type=str,
    )
    parser.add_argument(
        "--model_cls_file", 
        default='train/checkpoints/model.pth',
        type=str,
    )
    parser.add_argument(
        "--network_cls_file", 
        default="train/config/LAVR_class_config.py", 
        type=str,
    )
    parser.add_argument(
        "--config_file", 
        default="main/cls.yaml",
        type=str, 
    )
    args = parser.parse_args()
    return args


def inference(
    predictor: ResampledClassificationPredictor,
    hu_volume: np.ndarray,
    flow: np.ndarray,
    infar_mask: np.ndarray,
):
    is_art_cls = predictor.predict(hu_volume, flow, infar_mask)

    return is_art_cls


def load_scans(dcm_path):
    if dcm_path.endswith(".nii.gz"):
        sitk_img = sitk.ReadImage(dcm_path)
    else:
        reader = sitk.ImageSeriesReader()
        name = reader.GetGDCMSeriesFileNames(dcm_path)
        reader.SetFileNames(name)
        sitk_img = reader.Execute()
    return sitk_img


def main(input_dicom_path, input_flow_path, input_infar_path, output_path, gpu, args):
    if args.model_cls_file is not None and args.network_cls_file is not None and args.config_file is not None:
        model_segUrinary_vessel = ResampledClassificationModel(
            model_f=args.model_cls_file, network_f=args.network_cls_file, config_f=args.config_file,
        )
        predictor_segUrinary_vessel = ResampledClassificationPredictor(gpu=gpu, model=model_segUrinary_vessel,)
    else:
        print('tar:', args.model_path)
        with tarfile.open(args.model_path, "r") as tar:
            predictor_segUrinary_vessel = ResampledClassificationPredictor.build_predictor_from_tar(tar=tar, gpu=gpu)


    os.makedirs(output_path, exist_ok=True)
    patient_cls={}

    result = {"pid":[], "pred_cls": [], "pred_prob":[]}

    for patient_dir in tqdm(os.listdir(input_dicom_path)):
        print(patient_dir)
        pid = patient_dir.split('.nii.gz')[0]
        cls_dict={}
        cls_path = os.path.join(input_dicom_path, patient_dir)
        flow_path = os.path.join(input_flow_path, patient_dir)
        infar_path = os.path.join(input_infar_path, patient_dir)
        
        sitk_img = load_scans(cls_path)
        hu_volume = sitk.GetArrayFromImage(sitk_img)

        infar_itk = sitk.ReadImage(infar_path)
        infar_mask = sitk.GetArrayFromImage(infar_itk)

        flow_itk = sitk.ReadImage(flow_path)
        flow_mask = sitk.GetArrayFromImage(flow_itk)
        infar_mask[infar_mask > 1] = 0
        pred = inference(predictor_segUrinary_vessel, hu_volume, flow_mask, infar_mask)
        if pred >= 0.5:
            cls = 1
        else:
            cls = 0
        result["pid"].append(pid)
        result["pred_cls"].append(cls)
        result["pred_prob"].append(pred)
        
        patient_cls[patient_dir] = cls_dict
    
    df = pandas.DataFrame(result)
    df.to_excel(os.path.join(output_path, "pred_cls_test.xlsx"), index=False)


def read_cls_data(path: str):
    result = dict()
    with open(path) as f:
        for line in f.readlines():
            dicom, cls = line.split()
            result[int(cls)] = dicom
    return result


if __name__ == "__main__":
    args = parse_args()
    main(
        input_dicom_path=args.input_dicom_path,
        input_flow_path=args.input_flow_path,
        input_infar_path=args.input_infar_path,
        output_path=args.output_path,
        gpu=args.gpu,
        args=args,
    )
