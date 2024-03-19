from declearn.utils import deserialize_object, json_load
import pandas as pd

CONFIG_FILE = 'results/server/model_config.json'
STATE_FILE = 'results/server/model_state_24-03-19_18-07-33.json'
df = pd.read_csv('data/test_norm.csv')
features = [c for c in df.columns if c.split('_')[0] in ('thick','subVol','grayVol')]
X_test = df[features].to_numpy()
Y_test = df.age.to_numpy()

model = deserialize_object(CONFIG_FILE)
weights = json_load(STATE_FILE)
model.set_weights(weights)
preds = model._predict(X_test)
print(f"MAE = {abs(preds-Y_test).mean():.3f}")


