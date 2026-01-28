def add_arrays(arr1, arr2):
  if len(arr1) != len(arr2):
    return none
  return [x + y for x, y in zip(arr1, arr2)]
    print(add_arrays(arr1, arr2))        # Ожидаем: [6, 8, 10, 12]
    print(arr1)                          # Ожидаем: [1, 2, 3, 4] (оригинал не меняется)
    print(arr2)                          # Ожидаем: [5, 6, 7, 8]
    print(add_arrays(arr1, [1, 2, 3]))   # Ожидаем: None
