import logging
import os
import webdataset as wds
import csv
import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import json
import numpy as np
from PIL import Image

def resize_and_center_crop(image, target_width, target_height):
    if target_height == image.shape[0] and target_width == image.shape[1]:
        return image

    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)

def fix_nulls(s):
    for line in s:
        yield line.replace('\0', ' ')
        
def get_closest_ratio(height: float, width: float, ratios: list, buckets: list):
    """get the closest ratio in the buckets

    Args:
        height (float): video height
        width (float): video width
        ratios (list): video aspect ratio
        buckets (list): buckets generate by `generate_crop_size_list`

    Returns:
        the closest ratio in the buckets and the corresponding ratio
    """
    aspect_ratio = float(height) / float(width)
    diff_ratios = ratios - aspect_ratio

    if aspect_ratio >= 1:
        indices = [(index, x) for index, x in enumerate(diff_ratios) if x <= 0]
    else:
        indices = [(index, x) for index, x in enumerate(diff_ratios) if x > 0]

    closest_ratio_id = min(indices, key=lambda pair: abs(pair[1]))[0]
    closest_size = buckets[closest_ratio_id]
    closest_ratio = ratios[closest_ratio_id]

    return closest_size, closest_ratio

def get_closest_length(length, buckets: list):
    closest_length = buckets[0]
    for b in buckets: 
        if length-b>=0: closest_length = b
    return closest_length

def get_temporal_closest_ratio(video_length: float, height: float, width: float, ratios: list, temporal_buckets: list, buckets: list):
    # get closest temporal bucket
    assert video_length >= temporal_buckets[0], "video length is smaller than the first temporal bucket"
    closed_temporal_bucket = get_closest_length(video_length, temporal_buckets)
    # print(f'{video_length}-{closed_temporal_bucket}')
    # get closest spatial bucket
    aspect_ratio = float(height)/float(width)
    closest_ratio_id = np.abs(ratios - aspect_ratio).argmin()
    closest_ratio = min(ratios, key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return closed_temporal_bucket, buckets[closest_ratio_id], float(closest_ratio)

def generate_crop_size_list(base_size=256, patch_size=16, max_ratio=4.0):
    num_patches =  round((base_size / patch_size) ** 2)
    assert max_ratio >= 1.
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list


def generate_crop_size_w_temporal_list(base_size=256, patch_size=16, max_ratio=4.0, temporal_list=[]):
    assert len(temporal_list) > 0, "temporal_list is empty" 
    num_patches =  round((base_size / patch_size) ** 2)
    assert max_ratio >= 1.
    crop_size_list = []
    spatial_crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            spatial_crop_size_list.append((wp * patch_size, hp * patch_size))
            for temp_bucket in temporal_list:
                crop_size_list.append((int(temp_bucket), wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list, spatial_crop_size_list


def convert_csv_to_tar(csv_path, maxcount=2000000, save_dir_name='', vae_dir=''):
    if save_dir_name == '': 
        file_name = os.path.basename(csv_path).split('.')[0]+'_wds'
        save_dir_name = os.path.dirname(csv_path)
        save_dir_name = os.path.join(save_dir_name, file_name)
        
    pattern = f"{save_dir_name}/{file_name}_%03d.tar"
    print(f'Save tar to {save_dir_name}')
    os.makedirs(save_dir_name, exist_ok=True)
    with open(csv_path, 'rU', newline="\n") as csvfile:
        dataset = list(csv.DictReader(fix_nulls(csvfile)))
        length = len(dataset)
    print(f"Total number of record: {length}")
                
    with wds.ShardWriter(pattern, maxcount=maxcount) as of:
        for idx, info in tqdm.tqdm(enumerate(dataset)):
            try:
                # print(info)
                video_id = info['videoid']
                info['__key__'] = video_id
                if 'contentUrl' in info:
                    info['cos_url'] = info['contentUrl']
                elif 'cos_url' in info:
                    info['contentUrl'] = info['cos_url']
                of.write(info)
                # if idx>100: break
                # id = l.strip('\n')
                # image_path = f'/teg_amai_cq5/local_storage/0_public_datasets/SAM/images/sa_{id}.jpg'
                # caption_path = f'/teg_amai_cq5/local_storage/0_public_datasets/SAM/SA1B_caption/sa_{id}.txt'
                # key = f'sa_{id}'
                # with open(image_path, "rb") as stream: image = stream.read()
                # with open(caption_path, "rb") as stream: caption = stream.read()
                # tmp_dict = {'__key__': key, 'jpg': image, 'txt': caption}
                # of.write(tmp_dict)
                # if idx>100: break
                # print(image_path)
                # print(caption_path)
                # print(caption)
                # break
            except Exception as e:
                print(e)
    
def conver_csv_to_arrow(csv_path, save_dir_name='', vae_dir=''):
    if save_dir_name == '': 
        save_dir_name = os.path.dirname(csv_path)
        # save_dir_name = os.path.join(save_dir_name, file_name)
        
    os.makedirs(save_dir_name, exist_ok=True)
    with open(csv_path, 'rU', newline="\n") as csvfile:
        dataset = list(csv.DictReader(fix_nulls(csvfile)))
        length = len(dataset)
    print(f"Total number of record: {length}")
    
    df = pd.DataFrame(dataset)
    # 转换为pyarrow Table
    table = pa.Table.from_pandas(df)
    
    # 指定输出文件名
    # output_file = 'data.arrow'
    file_name = os.path.basename(csv_path).split('.')[0]+'.arrow'
    output_file = f'{save_dir_name}/{file_name}'
    print(f'Save arrow to {output_file}')

    # 写入文件
    writer = pa.ipc.new_file(output_file, table.schema)
    writer.write(table)
    writer.close()
    # pq.write_table(table, output_file)

    return output_file

# 制作多分辨率桶对应的json文件，每个arrow提前遍历一遍统计每个桶的数量以及桶的index
def convert_to_json(index_file, base_size=256, patch_size=16, use_latent_shape=False, 
                    multitemp=False, temporal_min_length=49, temporal_max_length=65, temporal_interval=8):
    with open(index_file) as f:
        d = json.load(f)
        assert 'arrow_list' in d, 'index_file must contain arrow_list'
        arrow_list = d['arrow_list']
        if 'cache_folder_list' in d: cache_folder_list = d['cache_folder_list']
        else: cache_folder_list = ["" for _ in arrow_list]
        
        if 'video_folder_list' in d: video_folder_list = d['video_folder_list']
        else: video_folder_list = ["" for _ in arrow_list]
        
        print(arrow_list)
        print(cache_folder_list)
        
        aspect_ratio_index_list = []
        aspect_ratios_nums_list = []
        for arrow_file, video_folder, cache_folder in zip(arrow_list, cache_folder_list, video_folder_list):
            table = pa.memory_map(arrow_file, "r")
            table = pa.ipc.RecordBatchFileReader(table).read_all()  # 打印第一条数据，0.0013秒
            
            if multitemp:
                # generate temporal bucket
                assert temporal_max_length>temporal_min_length, f"temporal_max_length must be larger than temporal_min_length, but got {temporal_max_length}, {temporal_min_length}"
                assert (temporal_min_length-1)%4==0, f"temporal_min_length-1 must be multiple of 4, but got {temporal_min_length}"
                assert (temporal_max_length-1)%4==0, f"temporal_max_length-1 must be multiple of 4, but got {temporal_max_length}"
                
                video_length_list = [fn+1 for fn in list(range(temporal_min_length-1, temporal_max_length-1, temporal_interval))]+[temporal_max_length]
                print(f"List of temporal length buckets: {video_length_list}")
                video_length_buckets = np.array(video_length_list)
            
            assert base_size in [128, 256, 480, 512, 720, 960], \
                f"bucket_hw_base_size must be in [128, 256, 480, 512, 720, 960], but got {base_size}"
                
            if multitemp:
                crop_size_list, spatial_crop_size_list = generate_crop_size_w_temporal_list(base_size, patch_size, temporal_list=video_length_buckets)
                print(f'crop_size_list: {crop_size_list}')
                aspect_ratios = np.array([round(float(h)/float(w), 5) for h, w in spatial_crop_size_list])
                print(f'aspect_ratios: {aspect_ratios}')
                temporal_aspect_ratio_index = {}
                temporal_aspect_ratios_nums = {}
                for temp_bucket in video_length_buckets:
                    if int(temp_bucket) not in temporal_aspect_ratio_index: temporal_aspect_ratio_index[int(temp_bucket)] = {}
                    if int(temp_bucket) not in temporal_aspect_ratios_nums: temporal_aspect_ratios_nums[int(temp_bucket)] = {}
                    for k in aspect_ratios:
                        temporal_aspect_ratio_index[int(temp_bucket)][float(k)] = []     # used for self.getitem
                        temporal_aspect_ratios_nums[int(temp_bucket)][float(k)] = 0      # used for batch-sampler
                        
                # scan the dataset for ratio static
                for idx in tqdm.tqdm(range(len(table))):
                    if use_latent_shape:
                        latent_shape = eval(table['latent_shape'][idx].as_py())
                        assert isinstance(latent_shape, list), "latent_shape must be list"
                        num_frames = latent_shape[-3]
                        height = latent_shape[-2]
                        width = latent_shape[-1]
                        num_frames = (num_frames-1)*4+1
                    else:
                        height = int(float(table['height'][idx].as_py()))
                        width = int(float(table['width'][idx].as_py()))
                        fps = float(table['fps'][idx].as_py())
                        duration = float(table['duration'][idx].as_py())
                        num_frames = int(fps*duration)
                    closest_temp, closest_size, closest_ratio = get_temporal_closest_ratio(num_frames, height, width, aspect_ratios, video_length_buckets, spatial_crop_size_list)
                    temporal_aspect_ratios_nums[closest_temp][float(closest_ratio)] += 1
                    # if len(aspect_ratio_index[closest_ratio]) < 10:
                    temporal_aspect_ratio_index[closest_temp][float(closest_ratio)].append(idx)
                # print(temporal_aspect_ratios_nums)
                # print(temporal_aspect_ratio_index)
                aspect_ratio_index_list.append(temporal_aspect_ratio_index)
                aspect_ratios_nums_list.append(temporal_aspect_ratios_nums)
                
            else:
                crop_size_list = generate_crop_size_list(base_size, patch_size)
                aspect_ratios = np.array([round(float(h)/float(w), 5) for h, w in crop_size_list])
                aspect_ratio_index = {}
                aspect_ratios_nums = {}
                for k in aspect_ratios:
                    aspect_ratio_index[float(k)] = []     # used for self.getitem
                    aspect_ratios_nums[float(k)] = 0      # used for batch-sampler
            
                # scan the dataset for ratio static
                cnt = 0
                for idx in tqdm.tqdm(range(len(table))):
                    if use_latent_shape:
                        latent_shape = eval(table['latent_shape'][idx].as_py())
                        assert isinstance(latent_shape, list), "latent_shape must be list"
                        num_frames = latent_shape[-3]
                        height = latent_shape[-2]
                        width = latent_shape[-1]
                    else:
                        height = int(float(table['height'][idx].as_py()))
                        width = int(float(table['width'][idx].as_py()))
        
                    _, closest_ratio = get_closest_ratio(height, width, aspect_ratios, crop_size_list)
                    aspect_ratios_nums[float(closest_ratio)] += 1
                    # if len(aspect_ratio_index[closest_ratio]) < 10:
                    aspect_ratio_index[float(closest_ratio)].append(idx)
                # print(aspect_ratios_nums)
                # print(aspect_ratio_index)
                aspect_ratio_index_list.append(aspect_ratio_index)
                aspect_ratios_nums_list.append(aspect_ratios_nums)
            
        d['base_size'] = base_size
        d['crop_size_list'] = crop_size_list
        if multitemp: d['temporal_buckets'] = video_length_buckets.tolist()
        d['aspect_ratios'] = aspect_ratios.tolist()
        d['aspect_ratios_nums_list'] = aspect_ratios_nums_list
        d['aspect_ratio_index_list'] = aspect_ratio_index_list
    with open(index_file, 'w') as f:
        json.dump(d, f, indent=4)
    
