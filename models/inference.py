#Credits: https://www.kaggle.com/code/voix97/jane-street-rmf-inference-nn-with-pytorch?scriptVersionId=212412682
from config import *
from AE import *

class CONFIG:
    seed = 42
    target_col = "responder_6"
    feature_cols = [f"feature_{idx:02d}" for idx in range(79)]+ [f"responder_{idx}_lag_1" for idx in range(9)]
    model_paths = [
        "/kaggle/input/sae_v1/pytorch/default/3"
    ]

N_folds = 2
models = []
for fold in range(N_folds):
    checkpoint_path = f"{CONFIG.model_paths[0]}/nn_{fold}.model.ckpt"
    model = AE.load_from_checkpoint(checkpoint_path)
    models.append(model.to("cuda:0"))

print(models[0])

gc.collect()

## Kaggle API
lags_ : pl.DataFrame | None = None
    
def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    global lags_
    if lags is not None:
        lags_ = lags

    predictions = test.select(
        'row_id',
        pl.lit(0.0).alias('responder_6'),
    )
    symbol_ids = test.select('symbol_id').to_numpy()[:, 0]

    if not lags is None:
        lags = lags.group_by(["date_id", "symbol_id"], maintain_order=True).last() # pick up last record of previous date
        test = test.join(lags, on=["date_id", "symbol_id"],  how="left")
        # print(a)
    else:
        test = test.with_columns(
            ( pl.lit(0.0).alias(f'responder_{idx}_lag_1') for idx in range(9) )
        )
    
    preds = np.zeros((test.shape[0],))
    # for i, model in enumerate(tqdm(models)):
    #     preds += model.predict(test[CONFIG.feature_cols].to_pandas()) / len(models)
    test_input = test[CONFIG.feature_cols].to_pandas()
    test_input = test_input.fillna(method = 'ffill').fillna(0)
    test_input = torch.FloatTensor(test_input.values).to("cuda:0")
    with torch.no_grad():
        for i, nn_model in enumerate(tqdm(models)):
            nn_model.eval()
            preds += nn_model(test_input).cpu().numpy() / len(models)
    print(f"predict> preds.shape =", preds.shape)
    
    predictions = \
    test.select('row_id').\
    with_columns(
        pl.Series(
            name   = 'responder_6', 
            values = np.clip(preds, a_min = -5, a_max = 5),
            dtype  = pl.Float64,
        )
    )

    # The predict function must return a DataFrame
    assert isinstance(predictions, pl.DataFrame | pd.DataFrame)
    # with columns 'row_id', 'responer_6'
    assert list(predictions.columns) == ['row_id', 'responder_6']
    # and as many rows as the test data.
    assert len(predictions) == len(test)

    return predictions

inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/jane-street-realtime-marketdata-forecasting/test.parquet',
            '/kaggle/input/jane-street-realtime-marketdata-forecasting/lags.parquet',
        )
    )