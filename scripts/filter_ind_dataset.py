lines = []
with open('ind_test.txt', 'r') as file:
    for line in file:
        lines.append(line.strip().split('\t'))

with open('ind_test2.txt', 'w') as file:
    for line in lines:
        if int(line[1]) in {3, 4, 5, 6}:
            continue
        file.write('\t'.join(line) + '\n')