## Training

To train a model, you must specify the `.py` file. The `.py` file contains all the configurations of the dataset and the model. We provide `.py` files under the [configs](../configs) directory. 

>Note: Before running the code, you will need to edit the `.py` file by setting the `path` argument as the correct root of the dataset.

    ```
    # single-gpu training
    python train.py --config configs/voxel/kitti/car.py
    # multi-gpu training
    # you will need to edit the `train.py` file by setting the `config` argument
    ./dist_train.sh
    ```
