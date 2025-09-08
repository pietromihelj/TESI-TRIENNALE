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

events_dir = ["D:/models/VAEEG_theta_z10/save","D:/models/VAEEG_delta_z8/save","D:/models/VAEEG_alpha_z12/save","D:/models/VAEEG_low_beta_z12/save","D:/models/VAEEG_high_beta_z12/save"]
mae_values = []
pearson_values = []
for dir in events_dir:
    ea = event_accumulator.EventAccumulator(events_dir)
    mae_values.append([e.value for e in ea.Scalars("MAE_error")])
    pearson_values.append([e.value for e in ea.Scalars("pearsonr")])