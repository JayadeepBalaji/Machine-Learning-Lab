inp = input("Enter the elements: ")
li = []
for s in inp.split(' '):
    li.append(int(s))
n = len(li)
mean = 0
for num in li:
    mean += num
mean /= n
print("Mean = ", mean)

frequencies = dict()
for num in li:
    frequencies[num] = li.count(num)
maxFrequency = max(frequencies.values())

list_of_modes = []
for num, frequency in frequencies.items():
    if frequency == maxFrequency:
        list_of_modes.append(num)

if (len(list_of_modes) == len(frequencies) and len(list_of_modes) != 1):
    print("No mode exists.")
else:
    print("Mode(s):", end = ' ')
    for mode in list_of_modes:
        print(mode, end = ' ')
    print()

li.sort()
if (n % 2 == 1):
    median = li[n // 2]
else:
    first = li[(n // 2) - 1]
    second = li[n // 2]
    median = (first + second) / 2

print("Median = ", median)

variance = 0
for num in li:
    variance += (num - mean) ** 2
variance /= n
print("Variance = ", variance)

print("Standard deviation = ", variance ** (1 / 2))
