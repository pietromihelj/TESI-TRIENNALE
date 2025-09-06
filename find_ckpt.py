from tensorboard.backend.event_processing import event_accumulator

events_dir = "D:/models/VAEEG_alpha_z12/save"

# carica il summary
ea = event_accumulator.EventAccumulator(events_dir)
ea.Reload()

# lista di tutti i tag salvati
print("Tags disponibili:", ea.Tags())

# esempio: leggo MAE e Pearson
mae_events = ea.Scalars("MAE_error")
pearson_events = ea.Scalars("pearsonr")

sorted_mae = sorted(mae_events, key= lambda e: e.value)
sorted_pears = sorted(pearson_events, key= lambda e: e.value, reverse=True)

for e in sorted_mae[:5]:  
    print(f"[MAE] Step: {e.step}, Value: {e.value}, Pearson associatp {[c.value for c in pearson_events if c.step == e.step]}")

for e in sorted_pears[:5]:
    print(f"[Pearson] Step: {e.step}, Value: {e.value}, Mae associato: {[c.value for c in mae_events if c.step == e.step]}")

