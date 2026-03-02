trainner = dict(type="Trainner", runner_config=dict(type="EpochBasedRunner"))
isotropy_spacing = [1, 1, 1]
patch_size = [96, 96, 96]

model = dict(
    type="Classification_Network",
    backbone=dict(type="CNNTrans", in_ch=4, channels=32, blocks=3),
    apply_sync_batchnorm=True,
    head=dict(type="Classification_Head"),
)

train_cfg = None
test_cfg = None

data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=1,
    shuffle=True,
    drop_last=False,
    dataloader=dict(type="SampleDataLoader", source_batch_size=3, source_thread_count=1, source_prefetch_count=1,),
    train=dict(
        type="ClassificationPidSampleDataset",
        root="processed_data",
        dst_list_file="processed_data/train.lst",
        patch_size=patch_size,
        isotropy_spacing=isotropy_spacing,
        rotation_prob=0.5,
        noise_prob=0.1,
        color_prob=0.6,
        rot_range=[5, 5, 5],
        spacing_range=0.05,
        shift_range=10,
        sample_frequent=1,
        whole_bright_aug=(1, 0.1, 0.1),

    ),

)

optimizer = dict(type="Adam", lr=5e-4, weight_decay=5e-4)
optimizer_config = {}

lr_config = dict(policy="step", warmup="linear", warmup_iters=10, warmup_ratio=1.0 / 3, step=[10, 25, 40], gamma=0.2)

checkpoint_config = dict(interval=1)

log_config = dict(interval=1, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")])

cudnn_benchmark = False
work_dir = "./checkpoints/model"
gpus = 4
find_unused_parameters = True
total_epochs = 100
autoscale_lr = None
validate = False
launcher = "pytorch"  # ['none', 'pytorch', 'slurm', 'mpi']
dist_params = dict(backend="nccl")
log_level = "INFO"
seed = None
deterministic = False
resume_from = None  
load_from = None
workflow = [("train", 1)]
