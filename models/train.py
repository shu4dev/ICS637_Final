#Credit: https://www.kaggle.com/code/voix97/jane-street-rmf-training-nn
from config import *
from dataset import *
from AE import *
gc.collect()

input_path = 'data/processed/jane-street-real-time-market-data-forecasting'
TRAINING = True
feature_names = [f"feature_{i:02d}" for i in range(79)] + [f"responder_{idx}_lag_1" for idx in range(9)]
label_name = 'responder_6'
weight_name = 'weight'
train_name =  "data/processed/jane-street-real-time-market-data-forecasting/nn_input_df_with_lags.pickle"
valid_name = "data/processedjane-street-real-time-market-data-forecasting/nn_valid_df_with_lags.pickle"
if TRAINING and not os.path.exists(train_name):
    df = pl.scan_parquet(f"{input_path}/training.parquet").collect().to_pandas()
    valid = pl.scan_parquet(f"{input_path}/validation.parquet").collect().to_pandas()
    df = pd.concat([df, valid]).reset_index(drop=True)# A trick to boost LB from 0.0045->0.005
    with open(train_name, "wb") as w:
        pickle.dump(df, w)
    with open(valid_name, "wb") as w:
        pickle.dump(valid, w)
elif TRAINING:
    with open(train_name, "rb") as r:
        df = pickle.load(r)
    with open(valid_name, "rb") as r:
        valid = pickle.load(r)

X_train = df[ feature_names ]
y_train = df[ label_name ]
w_train = df[ "weight" ]

X_valid = valid[ feature_names ]
y_valid = valid[ label_name ]
w_valid = valid[ "weight" ]

args = custom_args()
# checking device
device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() and args.usegpu else 'cpu')
accelerator = 'gpu' if torch.cuda.is_available() and args.usegpu else 'cpu'
loader_device = 'cpu'
# Initialize Data Module
df[feature_names] = df[feature_names].fillna(method = 'ffill').fillna(0)
valid[feature_names] = valid[feature_names].fillna(method = 'ffill').fillna(0)
data_module = DataModule(df, batch_size=args.bs, valid_df=valid, accelerator=loader_device)

pl.seed_everything(args.seed)
for fold in range(args.N_fold):
    data_module.setup(fold, args.N_fold)
    # Obtain input dimension
    input_dim = data_module.train_dataset.features.shape[1]
    # Initialize Model
    model = AE(
        num_columns=input_dim, 
        num_labels=1, 
        hidden_units=[48, 48, 448, 224, 224, 128], 
        dropout_rates=[0.03, 0.03, 0.4, 0.1, 0.4, 0.3, 0.2, 0.4]
    )
    # Initialize Logger
    if args.use_wandb:
        wandb_run = wandb.init(project=args.project, config=vars(args), reinit=True)
        logger = WandbLogger(experiment=wandb_run)
    else:
        logger = None
    # Initialize Callbacks
    early_stopping = EarlyStopping('val_loss', patience=args.patience, mode='min', verbose=False)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, verbose=False, filename=f"./models/nn_{fold}.model") 
    timer = Timer()
    # Initialize Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=[args.gpuid] if args.usegpu else None,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback, timer],
        enable_progress_bar=True
    )
    # Start Training
    trainer.fit(model, data_module.train_dataloader(args.loader_workers), data_module.val_dataloader(args.loader_workers))
    # You can find trained best model in your local path
    print(f'Fold-{fold} Training completed in {timer.time_elapsed("train"):.2f}s')