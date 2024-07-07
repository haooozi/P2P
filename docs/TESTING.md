## Testing

+ To test a trained model, specify the checkpoint location with `--resume_from` argument and set the `--phase` argument as `test`.

>Note: Before running the code, you will need to edit the `.py` file by setting the `path` argument as the correct root of the dataset.

    ```
    # single-gpu testing
    python test.py --config configs/voxel/kitti/car.py --load_from pretrained/voxel/kitti/car.pth
    # multi-gpu testing
    # you will need to edit the `test.py` file by setting the `config` and 'load_from' argument
    ./dist_test.sh
    ```
