## Steps to reproduce synthetic training data using the Habitat-Sim simulator

### Create a conda environment
```bash
conda create -n habitat python=3.8 habitat-sim=0.2.1 headless=2.0 -c aihabitat -c conda-forge
conda active habitat
conda install pytorch -c pytorch
pip install opencv-python tqdm
```

or (if you get the error `For headless systems, compile with --headless for EGL support`)
```
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim

conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat
pip install . -v
conda install pytorch -c pytorch
pip install opencv-python tqdm
```

### Download Habitat-Sim scenes
Download Habitat-Sim scenes:
- Download links can be found here: https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md
- We used scenes from the HM3D, habitat-test-scenes, ReplicaCad and ScanNet datasets.
- Please put the scenes in a directory `$SCENES_DIR` following the structure below:
(Note: the habitat-sim dataset installer may install an incompatible version for ReplicaCAD backed lighting.
The correct scene dataset can be dowloaded from Huggingface: `git clone git@hf.co:datasets/ai-habitat/ReplicaCAD_baked_lighting`).
```
$SCENES_DIR/
├──hm3d/
├──gibson/
├──habitat-test-scenes/
├──ReplicaCAD_baked_lighting/
└──scannet/
```

### Download renderings metadata 

Download metadata corresponding to each scene and extract them into a directory `$METADATA_DIR`
```bash
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/habitat_5views_v1_512x512_metadata.tar.gz
tar -xvzf habitat_5views_v1_512x512_metadata.tar.gz
```

### Render the scenes

Render the scenes in an output directory `$OUTPUT_DIR`
```bash
export METADATA_DIR="/path/to/habitat/5views_v1_512x512_metadata"
export SCENES_DIR="/path/to/habitat/data/scene_datasets/"
export OUTPUT_DIR="data/habitat_processed"
cd datasets_preprocess/habitat/
export PYTHONPATH=$(pwd)
# Print commandlines to generate images corresponding to each scene
python preprocess_habitat.py --scenes_dir=$SCENES_DIR --metadata_dir=$METADATA_DIR --output_dir=$OUTPUT_DIR
# Launch these commandlines in parallel e.g. using GNU-Parallel as follows:
python preprocess_habitat.py --scenes_dir=$SCENES_DIR --metadata_dir=$METADATA_DIR --output_dir=$OUTPUT_DIR | parallel -j 16
```

### Make a list of scenes

```bash
python find_scenes.py --root $OUTPUT_DIR
```