list1 = ['FP1', 'F7', 'O1', 'F3', 'C3', 'P3', 'FP2', 'F4', 'C4', 'P4', 'O2', 'F8', 'FZ', 'CZ', 'PZ', 'T7', 'P7', 'T8', 'P8']
list2 = ['FP1', 'FP2', 'F3', 'F4', 'FZ', 'F7', 'F8', 'P3', 'P4', 'PZ', 'C3', 'C4', 'CZ', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2']

# Canali in list1 ma non in list2
only_in_list1 = set(list1) - set(list2)

# Canali in list2 ma non in list1
only_in_list2 = set(list2) - set(list1)

print("Canali solo in list1:", only_in_list1)
print("Canali solo in list2:", only_in_list2)
print('ccc')
