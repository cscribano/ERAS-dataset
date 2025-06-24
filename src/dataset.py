# -*- coding: utf-8 -*-
# ---------------------

import faulthandler
faulthandler.enable()


import cv2
import torch
import random
import argparse
from pathlib import Path
import numpy as np
import rasterio as rio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.affinity import affine_transform

from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, Point

class ER4seg(Dataset):

	def __init__(self, data_root, input_size=256, mode="train"):
		# type: (str, int, str) -> None

		self.mode = mode

		# Configure input / target size and sclae
		self.input_size = input_size
		self.orig_size = 256
		self.scale = self.input_size / self.orig_size

		# List available annotations
		self.data = []
		data_root = Path(data_root)
		data_prov = [p.stem for p in data_root.iterdir() if p.is_dir()]

		# Reggio Emilia (RE) is isolated as test set
		assert "RE" in data_prov
		if mode == "train":
			data_prov.remove("RE")
		else:
			data_prov = ["RE"]

		for p in data_prov:

			# List available images for province
			prov_img_root = data_root / p / "tiff" / "s2"
			prov_imgs = [i for i in prov_img_root.iterdir() if i.suffix == ".tif"]

			# Check for corresponding labels
			prov_poly_root = data_root / p / "geojson"
			annot_data = [(p, prov_poly_root / (p.stem.split('_')[0]+'.geojson') )for p in prov_imgs]

			for a, _ in annot_data:
				assert a.is_file(), f"{a} is not a file"

			# Append
			self.data.extend(annot_data)


		# Data Augmentation
		res_tf = A.Resize(self.input_size, self.input_size)
		if self.mode == "train":
			self.pixel_tf = A.Compose([
				res_tf,
				# Pixel alteration
				A.ColorJitter(p=0.5, brightness=0.2, saturation=0.8, hue=0.15),
				A.OneOf([
					A.Blur(p=0.5),
					A.MedianBlur(p=0.5),
					A.GaussianBlur(p=0.5)
				]),
				ToTensorV2()
			])
		else:
			# val, test
			self.pixel_tf = A.Compose([
				res_tf,
				ToTensorV2()
			])

	def __getitem__(self, item):
		# type: (int) -> dict

		ret = {"original_size": (self.orig_size, self.orig_size)} # Original input size
		image_file, annot_file = self.data[item]

		# Random HFlip
		h_flip = random.random() < 0.5 if self.mode == "train" else False
		v_flip = random.random() < 0.5 if self.mode == "train" else False

		# Load Image
		with rio.open(str(image_file), 'r') as f:
			channels = [f.read(i) for i in [1,2,3]]  # [3,2,1] = BGR
			tile_rgb = np.stack(channels, axis=-1)
			image_afine = f.transform

		# Load Label
		polygons = gpd.read_file(str(annot_file))

		# Preprocess mask
		mask = self.prepare_mask(polygons, image_afine, h_flip, v_flip)

		# Preprocess image
		tile_rgb = np.nan_to_num(tile_rgb)
		tile_rgb = tile_rgb.clip(0, 255)

		# Random Flip
		if h_flip:
			tile_rgb = cv2.flip(tile_rgb, 1)
		if v_flip:
			tile_rgb = cv2.flip(tile_rgb, 0)

		im_crop = self.pixel_tf(image=tile_rgb)['image']
		instance_masks = torch.from_numpy(mask).type(torch.float32) # (B, H, W)

		return im_crop, instance_masks

	def prepare_mask(self, polygons, tf,  h_flip=False, v_flip=False):

		width, height = (self.orig_size, self.orig_size)
		annotations = []

		# Project to image coordinates
		tf = (~tf)
		tf_gp  = np.array([tf.a, tf.b, tf.d, tf.e, tf.xoff, tf.yoff])

		for n, contours in polygons.iterrows():

			pol = contours.geometry
			pol = affine_transform(pol, tf_gp) 		# Project to image coordinates

			# Pre-process polygon
			pol = pol.simplify(1.0)
			if not pol.is_valid:
				pol = pol.buffer(0)

			# Split Multiplygons
			if type(pol) == MultiPolygon or type(pol) == GeometryCollection:
				for p in pol.geoms:
					if type(p) != Polygon:
						continue

					assert type(p) == Polygon and p.is_valid
					if p.area > 5:
						annotations.append(p)

			elif type(pol) == Polygon:
				assert pol.is_valid  # and type(pol) == Polygon and
				if pol.area > 5: 
					annotations.append(pol)
			else:
				print(f"Invalid polygon type {type(pol)}")

		if len(annotations) == 0:
			mask = np.zeros((1, width, height), dtype=np.uint8)
			return mask, None

		# Rasterize
		annotations = sorted(annotations, key=lambda x: x.area, reverse=True)
		annotations = annotations[:150]

		masks = []
		for idx, poly in enumerate(annotations):
			mask = rasterize([poly], out_shape=(width, height)).astype(np.float32)
			box = poly.bounds
			w = box[2] - box[0]
			h = box[3] - box[1]
			if mask.sum() > 5 and w>3 and h>3:
				masks.append(mask)

		if len(masks) != 0:
			masks = np.stack(masks, axis=0)
		else:
			masks = np.zeros((0, width, height), dtype=np.float32)
			return masks

		# Flip
		if h_flip:
			masks = np.ascontiguousarray(masks[:, :, ::-1])
		if v_flip:
			masks = np.ascontiguousarray(masks[:, ::-1, :])

		return masks

	def __len__(self):
		return len(self.data)


def merge_masks(segmentation_masks):

	merged_mask = np.zeros(segmentation_masks[0].shape + (4,), dtype=np.float32)  # Initialize merged mask
	colors = plt.cm.get_cmap('hsv', len(segmentation_masks))  # Generate a set of colors

	for i, mask in enumerate(segmentation_masks):
		c = np.array(colors(i))
		c[-1] = 0.4
		color = np.array(c)  # Get RGB color from colormap
		merged_mask[mask > 0] = color  # Apply color to non-zero pixels in mask

	merged_mask = (merged_mask * 255.0).astype(np.uint8)
	merged_mask = cv2.cvtColor(merged_mask, cv2.COLOR_BGRA2BGR)
	return merged_mask


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	parser = argparse.ArgumentParser(description='ER4seg reading and visualization')
	parser.add_argument('--data_root', type=str, default='/home/carmelo/PROGETTI/ABACO/ERAS')
	args = parser.parse_args()
	
	ds = ER4seg(args.data_root, mode='test')
	print(len(ds))

	max_i = len(ds)
	x = 0
	for i in range(0, max_i):
		image, masks = ds[i]
		image = image.numpy().transpose(1,2,0).astype(np.uint8)

		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		masks = masks.numpy().astype(np.uint8)


		# Show
		color_mask = merge_masks(masks)
		overlay = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)

		cv2.imshow("result", image)
		cv2.imshow("mask", color_mask)

		cv2.imshow("overlay", overlay)

		while cv2.waitKey(0) != ord('q'):
			pass