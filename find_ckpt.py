from tensorboard.backend.event_processing import event_accumulator

"""

Questo codice serve a cercare il miglior checkpoint per mae e pearson, intorno ad un valore ottimo per lo smoothing

events_dir = "D:/models/VAEEG_alpha_z12/save"
ea = event_accumulator.EventAccumulator(events_dir)
ea.Reload()
mae_events = ea.Scalars("MAE_error")
pearson_events = ea.Scalars("pearsonr")
mae_filtered = [e for e in mae_events if 597 <= e.step <= 918]
pearson_filtered = [e for e in pearson_events if 597 <= e.step <= 918]
sorted_mae = sorted(mae_filtered, key= lambda e: e.value)
sorted_pears = sorted(pearson_filtered, key= lambda e: e.value, reverse=True)
print('RISULTATI PER LA BANDA ALPHA')
for e in sorted_mae[:5]:  
    print(f"[MAE] Step: {e.step}, Value: {e.value}, Pearson associatp {[c.value for c in pearson_events if c.step == e.step]}")

for e in sorted_pears[:5]:
    print(f"[Pearson] Step: {e.step}, Value: {e.value}, Mae associato: {[c.value for c in mae_events if c.step == e.step]}")
print('#######################################################################################################################################')

events_dir = "D:/models/VAEEG_delta_z8/save"
ea = event_accumulator.EventAccumulator(events_dir)
ea.Reload()
mae_events = ea.Scalars("MAE_error")
pearson_events = ea.Scalars("pearsonr")
mae_filtered = [e for e in mae_events if 998 <= e.step <= 1191]
pearson_filtered = [e for e in pearson_events if 998 <= e.step <= 1191]
sorted_mae = sorted(mae_filtered, key= lambda e: e.value)
sorted_pears = sorted(pearson_filtered, key= lambda e: e.value, reverse=True)
print('RISULTATI PER LA BANDA DELTA')
for e in sorted_mae[:5]:  
    print(f"[MAE] Step: {e.step}, Value: {e.value}, Pearson associatp {[c.value for c in pearson_events if c.step == e.step]}")

for e in sorted_pears[:5]:
    print(f"[Pearson] Step: {e.step}, Value: {e.value}, Mae associato: {[c.value for c in mae_events if c.step == e.step]}")

print('#######################################################################################################################################')
events_dir = "D:/models/VAEEG_theta_z10/save"
ea = event_accumulator.EventAccumulator(events_dir)
ea.Reload()
mae_events = ea.Scalars("MAE_error")
pearson_events = ea.Scalars("pearsonr")
mae_filtered = [e for e in mae_events if 600 <= e.step <= 960]
pearson_filtered = [e for e in pearson_events if 600 <= e.step <= 960]
sorted_mae = sorted(mae_filtered, key= lambda e: e.value)
sorted_pears = sorted(pearson_filtered, key= lambda e: e.value, reverse=True)
print('RISULTATI PER LA BANDA THETA')
for e in sorted_mae[:5]:  
    print(f"[MAE] Step: {e.step}, Value: {e.value}, Pearson associatp {[c.value for c in pearson_events if c.step == e.step]}")

for e in sorted_pears[:5]:
    print(f"[Pearson] Step: {e.step}, Value: {e.value}, Mae associato: {[c.value for c in mae_events if c.step == e.step]}")
print('#######################################################################################################################################')

events_dir = "D:/models/VAEEG_low_beta_z10/save"
ea = event_accumulator.EventAccumulator(events_dir)
ea.Reload()
mae_events = ea.Scalars("MAE_error")
pearson_events = ea.Scalars("pearsonr")
mae_filtered = [e for e in mae_events if 529 <= e.step <= 1045]
pearson_filtered = [e for e in pearson_events if 529 <= e.step <= 1045]
sorted_mae = sorted(mae_filtered, key= lambda e: e.value)
sorted_pears = sorted(pearson_filtered, key= lambda e: e.value, reverse=True)
print('RISULTATI PER LA BANDA LOW BETA')
for e in sorted_mae[:5]:  
    print(f"[MAE] Step: {e.step}, Value: {e.value}, Pearson associatp {[c.value for c in pearson_events if c.step == e.step]}")

for e in sorted_pears[:5]:
    print(f"[Pearson] Step: {e.step}, Value: {e.value}, Mae associato: {[c.value for c in mae_events if c.step == e.step]}")
print('#######################################################################################################################################')

events_dir = "D:/models/VAEEG_high_beta_z10/save"
ea = event_accumulator.EventAccumulator(events_dir)
ea.Reload()
mae_events = ea.Scalars("MAE_error")
pearson_events = ea.Scalars("pearsonr")
mae_filtered = [e for e in mae_events if 411 <= e.step <= 604]
pearson_filtered = [e for e in pearson_events if 411 <= e.step <= 604]
sorted_mae = sorted(mae_filtered, key= lambda e: e.value)
sorted_pears = sorted(pearson_filtered, key= lambda e: e.value, reverse=True)
print('RISULTATI PER LA BANDA HIGH BETA')
for e in sorted_mae[:5]:  
    print(f"[MAE] Step: {e.step}, Value: {e.value}, Pearson associatp {[c.value for c in pearson_events if c.step == e.step]}")

for e in sorted_pears[:5]:
    print(f"[Pearson] Step: {e.step}, Value: {e.value}, Mae associato: {[c.value for c in mae_events if c.step == e.step]}")

"""
import numpy as np
import matplotlib.pyplot as plt

def smooth(values, smooth_factor=0.9):
    """Exponential moving average smoothing (like TensorBoard)."""
    smoothed = []
    last = values[0]
    for v in values:
        last = smooth_factor * last + (1 - smooth_factor) * v
        smoothed.append(last)
    return np.array(smoothed)

"""
events_dir = ["D:/models/VAEEG_theta_z10/save","D:/models/VAEEG_delta_z8/save","D:/models/VAEEG_alpha_z12/save","D:/models/VAEEG_low_beta_z10/save","D:/models/VAEEG_high_beta_z10/save"]
bands = ['Theta', 'Delta', 'Alpha', 'Low Beta', 'High Beta']
mae_values = []
pearson_values = []
for dir in events_dir:
    ea = event_accumulator.EventAccumulator(dir)
    ea.Reload()
    mae_values.append([e.value for e in ea.Scalars("MAE_error")])
    pearson_values.append([e.value for e in ea.Scalars("pearsonr")])

mae_values_arrays = [np.array(l) for l in mae_values]
pearson_values_arrays = [np.array(l) for l in pearson_values]
np.save('mae.npy', np.array(mae_values_arrays, dtype=object), allow_pickle=True)
np.save('pearsons.npy', np.array(pearson_values_arrays, dtype=object), allow_pickle=True)
"""

pearson_values = np.load('C:/Users/Pietro/Desktop/TESI/TESI-TRIENNALE/pearsons.npy', allow_pickle=True)
mae_values = np.load('C:/Users/Pietro/Desktop/TESI/TESI-TRIENNALE/mae.npy', allow_pickle=True)
bands = ['Theta', 'Delta', 'Alpha', 'Low Beta', 'High Beta']

fig, axes = plt.subplots(2,5, figsize=(12,6))
steps = [np.arange(len(p)) for p in pearson_values]

handles, labels = [], []
for i, (mae, pear, band, step) in enumerate(zip(mae_values, pearson_values, bands, steps)):
    h1, = axes[0, i].plot(step, mae, label='MAE', color='lightgray')
    h2, = axes[0, i].plot(step, smooth(mae, 0.985), label='Smoothed MAE', color='blue')
    axes[0, i].set_title(band)
    axes[0, i].grid(True)
    h3, = axes[1, i].plot(step, pear, label='PCC', color='lightgray')
    h4, = axes[1, i].plot(step, smooth(pear, 0.985), label='Smoothed PCC', color='red')
    axes[1, i].grid(True)
    if i == 0:
        handles = [h1, h2, h3, h4]
        labels = ['MAE', 'Smoothed MAE', 'PCC', 'Smoothed PCC']

fig.text(0.04, 0.75, 'MAE', va='center', rotation='vertical', fontsize=12)
fig.text(0.04, 0.25, 'PCC', va='center', rotation='vertical', fontsize=12)
fig.text(0.5, 0.02, 'Step', ha='center', fontsize=12)
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=10)
plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.savefig('./results/VAEEG_Train_res.png')
plt.show()
