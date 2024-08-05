# Visual Localization with DUSt3R

## Dataset preparation

### CambridgeLandmarks

Each subscene should look like this:

```
Cambridge_Landmarks
├─ mapping
│   ├─ GreatCourt
│   │  └─ colmap/reconstruction
│   │     ├─ cameras.txt
│   │     ├─ images.txt
│   │     └─ points3D.txt
├─ kapture
│   ├─ GreatCourt
│   │  └─ query  # https://github.com/naver/kapture/blob/main/doc/datasets.adoc#cambridge-landmarks
│   ... 
├─ GreatCourt 
│   ├─ pairsfile/query
│   │     └─ AP-GeM-LM18_top50.txt  # https://github.com/naver/deep-image-retrieval/blob/master/dirtorch/extract_kapture.py followed by https://github.com/naver/kapture-localization/blob/main/tools/kapture_compute_image_pairs.py
│   ├─ seq1
│   ...
...
```

### 7Scenes
Each subscene should look like this:

```
7-scenes
├─ chess
│   ├─ mapping/  # https://github.com/naver/kapture/blob/main/doc/datasets.adoc#1-7-scenes
│   ├─ query/  # https://github.com/naver/kapture/blob/main/doc/datasets.adoc#1-7-scenes
│   └─ pairsfile/query/
│         └─ APGeM-LM18_top20.txt  # https://github.com/naver/deep-image-retrieval/blob/master/dirtorch/extract_kapture.py followed by https://github.com/naver/kapture-localization/blob/main/tools/kapture_compute_image_pairs.py
...
```

### Aachen-Day-Night

```
Aachen-Day-Night-v1.1
├─ mapping
│   ├─ colmap/reconstruction
│   │  ├─ cameras.txt
│   │  ├─ images.txt
│   │  └─ points3D.txt
├─ kapture
│   └─ query  # https://github.com/naver/kapture/blob/main/doc/datasets.adoc#2-aachen-day-night-v11
├─ images
│   ├─ db
│   ├─ query
│   └─ sequences
└─ pairsfile/query
    └─ fire_top50.txt  # https://github.com/naver/fire/blob/main/kapture_compute_pairs.py
```

### InLoc

```
InLoc
├─ mapping  # https://github.com/naver/kapture/blob/main/doc/datasets.adoc#6-inloc
├─ query    # https://github.com/naver/kapture/blob/main/doc/datasets.adoc#6-inloc
└─ pairsfile/query
    └─ pairs-query-netvlad40-temporal.txt  # https://github.com/cvg/Hierarchical-Localization/blob/master/pairs/inloc/pairs-query-netvlad40-temporal.txt
```

## Example Commands

With `visloc.py` you can run our visual localization experiments on Aachen-Day-Night, InLoc, Cambridge Landmarks and 7 Scenes.

```bash
# Aachen-Day-Night-v1.1:
# scene in 'day' 'night'
# scene can also be 'all'
python3 visloc.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt --dataset "VislocAachenDayNight('/path/to/prepared/Aachen-Day-Night-v1.1/', subscene='${scene}', pairsfile='fire_top50', topk=20)" --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/Aachen-Day-Night-v1.1/${scene}/loc

# InLoc
python3 visloc.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt --dataset "VislocInLoc('/path/to/prepared/InLoc/', pairsfile='pairs-query-netvlad40-temporal', topk=20)" --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/InLoc/loc


# 7-scenes:
# scene in 'chess' 'fire' 'heads' 'office' 'pumpkin' 'redkitchen' 'stairs'
python3 visloc.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt --dataset "VislocSevenScenes('/path/to/prepared/7-scenes/', subscene='${scene}', pairsfile='APGeM-LM18_top20', topk=1)" --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/7-scenes/${scene}/loc

# Cambridge Landmarks:
# scene in 'ShopFacade' 'GreatCourt' 'KingsCollege' 'OldHospital' 'StMarysChurch'
python3 visloc.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt --dataset "VislocCambridgeLandmarks('/path/to/prepared/Cambridge_Landmarks/', subscene='${scene}', pairsfile='APGeM-LM18_top50', topk=20)" --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/Cambridge_Landmarks/${scene}/loc

```
