import os
import shutil
import json
import glob

if __name__ == '__main__':
    json_path = r'D:\Projects\dust3r\data\0908\output\project-2-at-2024-09-08-19-57-30659d56.json'
    source_path = r'D:\Projects\dust3r\data\0908\output\project-2-at-2024-09-08-18-23-30659d56'
    target_dir = r'D:\Projects\dust3r\data\0908\output\mask'
    with open(json_path, 'r') as f:
        anno_info = json.load(f)
    for anno in anno_info:
        task_id = anno['id']
        file_upload = anno['file_upload']
        mask_source_file_name = glob.glob(os.path.join(source_path, 'task-{}-*.npy'.format(task_id)))[0]
        target_name = file_upload.split('-')[-1].replace('color', 'mask').replace('.jpg', '.npy')
        target_path = os.path.join(target_dir, target_name)
        print(target_name)
        shutil.copy(src=mask_source_file_name, dst=target_path)
